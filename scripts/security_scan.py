#!/usr/bin/env python3
# security_scan.py: Trivy over the image this fork publishes and its uv.lock, reported as one open
# GitHub issue labelled `security-scan` on drewlindsey/speaches. Fork-only, like fork-image.yaml:
# don't cherry-pick it upstream. .github/workflows/security-scan.yaml runs it every Monday;
# .github/security-scan.md says what it scans and how to act on the issue.
#
#   scripts/security_scan.py --list                   print the targets and stop
#   scripts/security_scan.py --dry-run                scan everything, print the issue body, touch no issue
#   scripts/security_scan.py --dry-run --only REGEX   scan only the targets whose name matches
#   scripts/security_scan.py                          scan, then open, update or close the issue (CI)
#
# Needs git, Docker and python3 (standard library only). Trivy runs from its pinned image, so the
# version is the same on a Mac and in CI. Issue mode also needs gh with issues: write on REPO, and
# it only ever talks to REPO, never to upstream.
#
# The same script runs in homelab, chloe-voice and this fork. Only the settings block below
# differs between them.
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

# --- Settings for this repo ---------------------------------------------------------------------

REPO = "drewlindsey/speaches"
DOC = ".github/security-scan.md"
# Off here: the compose files run upstream's images (ghcr.io/speaches-ai), not this fork's.
SCAN_COMPOSE = False
# Off here: the Dockerfile's base is a build arg, and the image it builds is scanned below.
SCAN_FROM_LINES = False
# The CPU image fork-image.yaml publishes from master (homelab's services/speaches pins a digest of it).
EXTRA_IMAGES: list[tuple[str, str]] = [
    ("ghcr.io/drewlindsey/speaches:master-cpu", ".github/workflows/fork-image.yaml"),
]
# Images to build from a context folder and scan as built, as (name, context).
BUILT_IMAGES: list[tuple[str, str]] = []
# Also scan the committed files for secrets.
SCAN_TREE_SECRETS = False


# --- The same in every repo ---------------------------------------------------------------------

LABEL = "security-scan"
# The newest Trivy release at least 7 days old when pinned (v0.76.2, 2026-11-05), by index digest.
TRIVY_IMAGE = "aquasec/trivy:0.76.2@sha256:8f6bb947dbc528902e11f2f5be52f7b519f6a39d5d148f41166f2040499768fb"
SEVERITY = "HIGH,CRITICAL"
SEVERITIES = ("CRITICAL", "HIGH")
# The hosts are amd64. Without this a Mac would scan an image's arm64 variant instead.
PLATFORM = "linux/amd64"
COMPOSE = re.compile(r"(^|/)(docker-)?compose[^/]*\.ya?ml$")
DOCKERFILE = re.compile(r"(^|/)Dockerfile[^/]*$")
LOCKFILE = re.compile(
    r"(^|/)(uv\.lock|poetry\.lock|Pipfile\.lock|requirements[^/]*\.(txt|lock)|pnpm-lock\.yaml|package-lock\.json|yarn\.lock)$"
)
SKIP = re.compile(r"(^|/)(node_modules|tests?/fixtures)/")
# Trivy reads pip requirements only from files named requirements.txt; this adds the .lock names.
PIP_FILES = r"pip:requirements[^/]*\.(txt|lock)$"
# A CUDA image is several GB: minutes on a runner, but most of an hour over a home connection.
TRIVY_TIMEOUT = "90m"
STEP_TIMEOUT = 95 * 60  # seconds, for one build or one scan
BODY_LIMIT = 60_000  # GitHub's issue body limit is 65,536 characters
ORDER = ("image", "base image", "built image", "lockfile", "tree")


class ScanError(Exception):
    pass


class Target:
    def __init__(self, kind: str, name: str, ref: str, sources: list[str]) -> None:
        self.kind = kind
        self.name = name
        self.ref = ref
        self.sources = sources
        self.error = ""
        self.packages = 0
        self.notes: list[str] = []
        self.vulns: list[dict[str, str]] = []
        self.secrets: list[dict[str, str]] = []

    def count(self, severity: str) -> int:
        return sum(1 for f in self.vulns + self.secrets if f["severity"] == severity)


def tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"{name} is not on PATH")
    return path


def run(cmd: list[str], stdin: str | None = None) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, input=stdin, check=False)
    if p.returncode != 0:
        raise SystemExit(f"{Path(cmd[0]).name} {cmd[1]} failed: {p.stderr.strip() or p.stdout.strip()}")
    return p.stdout


def last_lines(text: str, n: int = 3) -> str:
    lines = [re.sub(r"^\S+\s+(FATAL|ERROR)\s+(Fatal error\s+)?", "", line).strip() for line in text.splitlines()]
    message = " / ".join([line for line in lines if line][-n:]) or "no output"
    message = re.sub(r"(https?://[^/\s\"]+)[^\s\"]*", r"\1/...", message)  # a signed blob URL is long
    # Trivy's error is one long chain ("run error: image scan error: ... : <cause>"); the cause is last.
    return message if len(message) <= 400 else f"{message[:100]} ... {message[-280:]}"


# --- Finding the targets ------------------------------------------------------------------------


def has_sibling_key(lines: list[str], i: int, indent: int, key: str) -> bool:
    for step in (-1, 1):
        j = i + step
        while 0 <= j < len(lines):
            text = lines[j].strip()
            if text and not text.startswith("#"):
                depth = len(lines[j]) - len(lines[j].lstrip())
                if depth < indent:
                    break
                if depth == indent and re.match(rf"{key}\s*:", text):
                    return True
            j += step
    return False


def compose_images(text: str) -> list[str]:
    lines = text.splitlines()
    matches = [(i, re.match(r"^(\s*)image:\s*[\"']?([^\s\"'#]+)", line)) for i, line in enumerate(lines)]
    return [m[2] for i, m in matches if m and "$" not in m[2] and not has_sibling_key(lines, i, len(m[1]), "build")]


def from_images(text: str) -> list[str]:
    stages: set[str] = set()
    refs = []
    for line in text.splitlines():
        m = re.match(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)(?:\s+AS\s+(\S+))?", line, re.IGNORECASE)
        if not m:
            continue
        if "$" not in m[1] and m[1].lower() != "scratch" and m[1].lower() not in stages:
            refs.append(m[1])
        if m[2]:
            stages.add(m[2].lower())
    return refs


def image_key(ref: str) -> str:
    name, _, digest = ref.partition("@")
    if digest and ":" in name.rsplit("/", 1)[-1]:
        name = name.rsplit(":", 1)[0]  # beside a digest, the tag is only a label
    name = re.sub(r"^(docker\.io/)?(library/)?", "", name)
    return f"{name}@{digest}" if digest else name


def discover(root: Path) -> list[Target]:
    files = sorted(f for f in run([tool("git"), "-C", str(root), "ls-files", "-z"]).split("\0") if f)
    files = [f for f in files if not SKIP.search(f) and (root / f).is_file()]
    images: dict[str, Target] = {}

    def add(kind: str, ref: str, source: str) -> None:
        key = image_key(ref)
        if key in images:
            if source not in images[key].sources:
                images[key].sources.append(source)
            return
        images[key] = Target(kind, re.sub(r"(@sha256:[0-9a-f]{12})[0-9a-f]+", r"\1", ref), ref, [source])
        if "@sha256:" not in ref:
            images[key].notes.append("a tag, not a digest")

    for f in files:
        if SCAN_COMPOSE and COMPOSE.search(f):
            for ref in compose_images((root / f).read_text(encoding="utf-8", errors="replace")):
                add("image", ref, f)
    for f in files:
        if SCAN_FROM_LINES and DOCKERFILE.search(f):
            for ref in from_images((root / f).read_text(encoding="utf-8", errors="replace")):
                add("base image", ref, f)
    for ref, source in EXTRA_IMAGES:
        add("image", ref, source)
    targets = list(images.values())
    targets += [Target("built image", f"{name} (built from {ctx}/)", ctx, [ctx]) for name, ctx in BUILT_IMAGES]
    targets += [Target("lockfile", f, f, []) for f in files if LOCKFILE.search(f)]
    if SCAN_TREE_SECRETS:
        targets.append(Target("tree", "committed files (secrets)", "HEAD", []))
    return sorted(targets, key=lambda t: (ORDER.index(t.kind), t.name))


# --- Scanning -----------------------------------------------------------------------------------


class Trivy:
    # Each container is named, so one that outlives its docker CLI (a timeout, Ctrl-C) can be removed.
    PREFIX = f"security-scan-{os.getpid()}-"

    def __init__(self, root: Path, cache: Path, work: Path) -> None:
        self.docker = tool("docker")
        self.runs = 0
        self.cmd = [
            *(self.docker, "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}"),
            *("-v", f"{cache}:/cache", "-v", f"{root}:/src:ro", "-v", f"{work}:/work"),
            # HTTP/1.1: over HTTP/2, the blob stores behind GHCR and MCR reset a long layer download
            # partway ("stream error: CANCEL" or "PROTOCOL_ERROR"), which fails the whole image.
            *("-e", "GODEBUG=http2client=0"),
        ]
        self.flags = [
            *("--cache-dir", "/cache", "--quiet", "--format", "json"),
            *("--severity", SEVERITY, "--ignore-unfixed", "--timeout", TRIVY_TIMEOUT),
            *("--disable-telemetry", "--skip-version-check"),
        ]
        for name in (".trivyignore.yaml", ".trivyignore"):
            if (root / name).is_file():
                self.flags += ["--ignorefile", f"/src/{name}"]
                break

    def scan(self, subcommand: str, args: list[str]) -> dict:
        self.runs += 1
        name = f"{self.PREFIX}{self.runs}"
        try:
            p = subprocess.run(
                [*self.cmd, "--name", name, TRIVY_IMAGE, subcommand, *self.flags, *args],
                capture_output=True,
                text=True,
                timeout=STEP_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self.stop()
            raise ScanError(f"no result after {STEP_TIMEOUT // 60} minutes") from None
        if p.returncode != 0:  # with no message, the code says what happened (137: killed, often for memory)
            raise ScanError(
                last_lines(p.stderr) if p.stderr.strip() else f"Trivy exited {p.returncode} with no message"
            )
        try:
            return json.loads(p.stdout)
        except ValueError:
            raise ScanError(f"unreadable output: {last_lines(p.stderr)}") from None

    def stop(self) -> None:
        ps = [self.docker, "ps", "-aq", "--filter", f"name=^{self.PREFIX}"]
        ids = subprocess.run(ps, capture_output=True, text=True, check=False).stdout.split()
        if ids:
            subprocess.run([self.docker, "rm", "-f", *ids], capture_output=True, check=False)

    def version(self) -> str:
        cmd = [*self.cmd, TRIVY_IMAGE, "version", "--cache-dir", "/cache", "--format", "json"]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        try:
            info = json.loads(p.stdout)
        except ValueError:
            return "Trivy"
        db = (info.get("VulnerabilityDB") or {}).get("UpdatedAt", "")[:10]
        return f"Trivy {info.get('Version', '?')}" + (f", database of {db}" if db else "")


def advisory_url(vuln_id: str) -> str:
    if vuln_id.startswith("GHSA-"):
        return f"https://github.com/advisories/{vuln_id}"
    if vuln_id.startswith("CVE-"):
        return f"https://nvd.nist.gov/vuln/detail/{vuln_id}"
    return ""


def collect(t: Target, report: dict) -> None:
    os_info = (report.get("Metadata") or {}).get("OS") or {}
    if os_info.get("EOSL"):
        t.notes.append(f"{os_info.get('Family')} {os_info.get('Name')} is end of life")
    seen = set()
    for result in report.get("Results") or []:
        t.packages += len(result.get("Packages") or [])
        for v in result.get("Vulnerabilities") or []:
            row = {
                "severity": v.get("Severity", "UNKNOWN"),
                "package": v.get("PkgName", "?"),
                "installed": v.get("InstalledVersion", ""),
                "fixed": v.get("FixedVersion", ""),
                "id": v.get("VulnerabilityID", "?"),
            }
            key = (row["package"], row["installed"], row["id"])
            if key not in seen:  # the same package in several binaries or layers is one row
                seen.add(key)
                row["url"] = v.get("PrimaryURL") or advisory_url(row["id"])
                t.vulns.append(row)
        # A secret's matched text is never kept, so it can't reach the issue.
        t.secrets += [
            {
                "severity": s.get("Severity", "UNKNOWN"),
                "file": result.get("Target", "?"),
                "line": str(s.get("StartLine", "")),
                "rule": s.get("Title") or s.get("RuleID", "?"),
            }
            for s in result.get("Secrets") or []
        ]
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    t.vulns.sort(key=lambda r: (rank.get(r["severity"], 9), r["package"], r["id"]))
    t.secrets.sort(key=lambda r: (rank.get(r["severity"], 9), r["file"]))


def scan_built(t: Target, name: str, root: Path, trivy: Trivy, work: Path) -> None:
    tag = f"security-scan/{name}:scan"
    docker = tool("docker")
    try:
        p = subprocess.run(
            [docker, "build", "--pull", "--progress", "plain", "-t", tag, str(root / t.ref)],
            capture_output=True,
            text=True,
            timeout=STEP_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ScanError(f"the build took over {STEP_TIMEOUT // 60} minutes") from None
    if p.returncode != 0:
        raise ScanError(f"the build failed: {last_lines(p.stderr, 4)}")
    tar = work / f"{name}.tar"
    try:
        run([docker, "save", "-o", str(tar), tag])
        collect(t, trivy.scan("image", ["--input", f"/work/{tar.name}", "--scanners", "vuln,secret"]))
    finally:
        tar.unlink(missing_ok=True)
        subprocess.run([docker, "image", "rm", "-f", tag], capture_output=True, check=False)
        if os.environ.get("GITHUB_ACTIONS") == "true":  # a runner's disk only; never a Mac's build cache
            subprocess.run([docker, "builder", "prune", "-af"], capture_output=True, check=False)


def scan_tree(t: Target, root: Path, trivy: Trivy, work: Path) -> None:
    tree = work / "tree"
    tree.mkdir()
    try:
        # Committed files only: a checkout's gitignored secrets/ and .env are never read.
        with subprocess.Popen([tool("git"), "-C", str(root), "archive", "HEAD"], stdout=subprocess.PIPE) as git:
            subprocess.run([tool("tar"), "-x", "-C", str(tree)], stdin=git.stdout, check=True)
        collect(t, trivy.scan("fs", ["--scanners", "secret", "/work/tree"]))
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def scan_remote(trivy: Trivy, args: list[str], attempts: int = 3) -> dict:
    # A registry can drop a multi-GB download partway, mostly over a slow link: try again before
    # calling it a scan error.
    for attempt in range(1, attempts + 1):
        try:
            return trivy.scan("image", args)
        except ScanError as e:
            if attempt == attempts or re.search(r"deadline exceeded|no result after", str(e)):
                raise
            print(f"    try {attempt} failed, trying again: {e}", file=sys.stderr, flush=True)
    raise AssertionError


def scan(t: Target, root: Path, trivy: Trivy, work: Path) -> None:
    if t.kind == "lockfile":
        collect(t, trivy.scan("fs", ["--scanners", "vuln", "--file-patterns", PIP_FILES, f"/src/{t.ref}"]))
    elif t.kind == "built image":
        scan_built(t, next(name for name, ctx in BUILT_IMAGES if ctx == t.ref), root, trivy, work)
    elif t.kind == "tree":
        scan_tree(t, root, trivy, work)
    else:
        # Straight from the registry: nothing is pulled into Docker, so even a CUDA image needs no
        # more disk than Trivy's cache, and every scan starts from the same free space.
        args = ["--image-src", "remote", "--platform", PLATFORM, "--scanners", "vuln", t.ref]
        collect(t, scan_remote(trivy, args))
    # The count shows up a scan that read nothing, such as a lockfile Trivy can't parse.
    if t.kind != "tree":
        t.notes.append(f"{t.packages} packages" if t.packages else "Trivy read no packages")


# --- The issue body -----------------------------------------------------------------------------


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip() or " "


def totals(targets: list[Target]) -> dict[str, int]:
    return {
        "critical": sum(t.count("CRITICAL") for t in targets),
        "high": sum(t.count("HIGH") for t in targets),
        "errors": sum(1 for t in targets if t.error),
    }


def summary(n: dict[str, int]) -> str:
    text = f"{n['critical']} critical, {n['high']} high"
    if n["errors"]:
        text += f", {n['errors']} scan error" + ("s" if n["errors"] != 1 else "")
    return text


def fingerprint(targets: list[Target]) -> str:
    keys = sorted(
        [f"{t.ref}|{v['id']}|{v['package']}|{v['installed']}" for t in targets for v in t.vulns]
        + [f"{t.ref}|{s['rule']}|{s['file']}" for t in targets for s in t.secrets]
        + [f"{t.ref}|error" for t in targets if t.error]
    )
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def section(t: Target, max_rows: int | None, full_note: str) -> list[str]:
    out = ["", f"### `{cell(t.name)}`", ""]
    if t.sources:
        out += ["From " + ", ".join(f"`{s}`" for s in t.sources) + ".", ""]
    shown = 0
    if t.vulns:
        rows = t.vulns if max_rows is None else t.vulns[:max_rows]
        shown += len(rows)
        out += ["| Package | Installed | Fixed in | Advisory | Severity |", "|---|---|---|---|---|"]
        for v in rows:
            advisory = f"[{cell(v['id'])}]({v['url']})" if v["url"] else cell(v["id"])
            out.append(
                f"| {cell(v['package'])} | {cell(v['installed'])} | {cell(v['fixed'])} | {advisory} | {v['severity']} |"
            )
    if t.secrets:
        rows = t.secrets if max_rows is None else t.secrets[:max_rows]
        shown += len(rows)
        out += [""] if t.vulns else []
        out += ["| Secret | File | Line | Severity |", "|---|---|---|---|"]
        out += [f"| {cell(s['rule'])} | `{cell(s['file'])}` | {s['line']} | {s['severity']} |" for s in rows]
    hidden = len(t.vulns) + len(t.secrets) - shown
    if hidden:
        out += ["", f"...and {hidden} more. {full_note}"]
    return out


def render(targets: list[Target], ctx: dict[str, str], max_rows: int | None = None) -> str:
    n = totals(targets)
    dirty = [t for t in targets if t.vulns or t.secrets]
    out = [
        f"<!-- {LABEL} fingerprint={fingerprint(targets)} critical={n['critical']} high={n['high']} errors={n['errors']} -->"
    ]
    if dirty:
        headline = f"**{n['critical']} critical and {n['high']} high** findings with a fix available, in {len(dirty)} of {len(targets) - n['errors']} targets."
    else:
        headline = f"No HIGH or CRITICAL findings with a fix available in {len(targets) - n['errors']} targets."
    if n["errors"]:
        headline += f" **{n['errors']} couldn't be scanned**, so their findings are unknown."
    out += [
        headline,
        "",
        f"Scanned {ctx['date']} {ctx['run']}, with {ctx['trivy']} (`--severity {SEVERITY} --ignore-unfixed`). "
        f"What to do with a finding: `{DOC}`.",
        "",
        "| Target | Kind | Critical | High | Note |",
        "|---|---|--:|--:|---|",
    ]
    for t in targets:
        if t.error:
            out.append(f"| `{cell(t.name)}` | {t.kind} | | | **scan error** |")
        else:
            out.append(
                f"| `{cell(t.name)}` | {t.kind} | {t.count('CRITICAL')} | {t.count('HIGH')} | {cell('; '.join(t.notes))} |"
            )
    errors = [t for t in targets if t.error]
    if errors:
        out += ["", "### Scan errors", ""]
        out += [f"- `{cell(t.name)}`: {t.error}" for t in errors]
    for t in dirty:
        out += section(t, max_rows, ctx["full"])
    return "\n".join(out) + "\n"


def fit(targets: list[Target], ctx: dict[str, str]) -> str:
    body = ""
    for max_rows in (None, 200, 100, 50, 25, 10, 5, 0):
        body = render(targets, ctx, max_rows)
        if len(body) <= BODY_LIMIT:
            return body
    return body[: BODY_LIMIT - 200] + f"\n\n...cut to fit GitHub's limit. {ctx['full']}\n"


# --- The issue ----------------------------------------------------------------------------------


def gh(*args: str, stdin: str | None = None) -> str:
    return run([tool("gh"), *args, "--repo", REPO], stdin=stdin)


def previous(body: str) -> dict[str, int] | None:
    m = re.search(rf"<!-- {LABEL} fingerprint=(\w+) critical=(\d+) high=(\d+) errors=(\d+) -->", body or "")
    if not m:
        return None
    return {"fingerprint": int(m[1], 16), "critical": int(m[2]), "high": int(m[3]), "errors": int(m[4])}


def update_issue(targets: list[Target], body: str, ctx: dict[str, str]) -> None:
    n = totals(targets)
    labels = json.loads(gh("label", "list", "--json", "name", "--limit", "500"))
    if LABEL not in {label["name"] for label in labels}:
        gh("label", "create", LABEL, "--color", "B60205", "--description", "Findings from the weekly security scan")
    issues = json.loads(gh("issue", "list", "--label", LABEL, "--state", "open", "--json", "number,body"))
    issue = min(issues, key=lambda i: i["number"]) if issues else None
    title = f"Security scan: {summary(n)}"
    if n["critical"] or n["high"] or n["errors"]:
        if issue is None:
            url = gh("issue", "create", "--title", title, "--label", LABEL, "--body-file", "-", stdin=body)
            print(f"opened {url.strip()}", file=sys.stderr)
            return
        number = str(issue["number"])
        before = previous(issue["body"])
        gh("issue", "edit", number, "--title", title, "--body-file", "-", stdin=body)
        if before and before["fingerprint"] == int(fingerprint(targets), 16):
            print(f"updated #{number}: same findings", file=sys.stderr)
            return
        was = f" (was {summary(before)})" if before else ""
        gh("issue", "comment", number, "--body", f"Findings changed on {ctx['date']}: {summary(n)}{was}.{ctx['link']}")
        print(f"updated #{number}: findings changed", file=sys.stderr)
    elif issue is not None:
        gh("issue", "close", str(issue["number"]), "--comment", f"Clean as of {ctx['date']}.{ctx['link']}")
        print(f"closed #{issue['number']}", file=sys.stderr)
    else:
        print("clean, and no issue open", file=sys.stderr)


# --- Main ---------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Trivy over this repo's images and lockfiles, into one issue.")
    parser.add_argument("--dry-run", action="store_true", help="print the issue body and touch no issue")
    parser.add_argument("--list", action="store_true", help="print the targets and stop")
    parser.add_argument("--only", metavar="REGEX", help="scan only the targets that match (with --dry-run)")
    args = parser.parse_args()
    if args.only and not (args.dry_run or args.list):
        parser.error("--only scans part of the repo, so it can't update the issue: add --dry-run")

    here = Path(__file__).resolve().parent
    root = Path(run([tool("git"), "-C", str(here), "rev-parse", "--show-toplevel"]).strip())
    targets = discover(root)
    if args.only:
        targets = [t for t in targets if re.search(args.only, t.name) or re.search(args.only, t.ref)]
    if args.list:
        for t in targets:
            print(f"{t.kind:<12} {t.ref}" + (f"   ({', '.join(t.sources)})" if t.sources else ""))
        return 0
    if not targets:
        raise SystemExit("no targets")

    docker = tool("docker")
    if subprocess.run([docker, "image", "inspect", TRIVY_IMAGE], capture_output=True, check=False).returncode:
        run([docker, "pull", "-q", TRIVY_IMAGE])
    cache_env = os.environ.get("SECURITY_SCAN_CACHE")
    cache = Path(cache_env).expanduser() if cache_env else Path(tempfile.mkdtemp(prefix="security-scan-cache-"))
    cache.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="security-scan-"))
    trivy = Trivy(root, cache.resolve(), work.resolve())
    # A SIGTERM (a cancelled job, `kill`) still runs the cleanup below, as Ctrl-C does.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        for i, t in enumerate(targets, 1):
            started = time.monotonic()
            print(f"[{i}/{len(targets)}] {t.kind} {t.ref}", file=sys.stderr, flush=True)
            try:
                scan(t, root, trivy, work)
                result = f"{t.count('CRITICAL')} critical, {t.count('HIGH')} high"
            except ScanError as e:
                t.error = str(e)
                result = f"SCAN ERROR: {t.error}"
            free = shutil.disk_usage(work).free / 2**30
            print(f"    {result} ({time.monotonic() - started:.0f}s, {free:.1f} GiB free)", file=sys.stderr, flush=True)
        trivy_version = trivy.version()
    finally:
        trivy.stop()
        shutil.rmtree(work, ignore_errors=True)
        if not cache_env:
            shutil.rmtree(cache, ignore_errors=True)

    run_url = os.environ.get("RUN_URL")
    ctx = {
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "run": f"in [this run]({run_url})" if run_url else "by hand",
        "link": f" [Run]({run_url})." if run_url else "",
        "trivy": trivy_version,
        "full": "The run's summary page has every row." if run_url else "`--dry-run --only <target>` prints them.",
    }
    body = fit(targets, ctx)
    print(body)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:  # the run's page has the whole report, even when the issue body had to be cut
        full = render(targets, ctx)
        with Path(summary_file).open("a") as f:
            f.write(full if len(full) < 1_000_000 else body)
    if not args.dry_run:
        update_issue(targets, body, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
