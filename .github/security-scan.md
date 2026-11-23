# Security scan (fork-only)

Every Monday at 13:17 UTC (07:17 Central in winter), and whenever it's run by hand,
[`workflows/security-scan.yaml`](workflows/security-scan.yaml) runs
[`scripts/security_scan.py`](../scripts/security_scan.py), and keeps one issue on this fork,
labelled `security-scan`, up to date. Like `fork-image.yaml`, it's fork-only: don't cherry-pick it
upstream, and the job only runs in `drewlindsey/speaches`.

## What it scans

- **The image this fork publishes:** `ghcr.io/drewlindsey/speaches:master-cpu`, which
  `fork-image.yaml` pushes from every merge to `master`. Trivy reads it straight from GHCR as
  `linux/amd64`, so it never lands on the runner's disk.
- **`uv.lock`:** the lock the image installs from (`uv sync --frozen --no-dev`).

It reports HIGH and CRITICAL findings that have a fix upstream (`--ignore-unfixed`): a finding
with no fix is nothing a change here can act on, and an issue that never closes gets ignored. The
compose files and the Dockerfile's base images aren't scanned: the compose files run upstream's
images, and the base is a build arg whose result is the image above.

## The issue

- **Findings or scan errors, no issue open:** it opens one.
- **An issue open:** it rewrites the body, and comments when the findings changed since the last
  run.
- **Clean:** it comments "Clean as of <date>" and closes the issue.

The body has a table of counts per target, then one table per target with findings: package,
installed version, fixed version, advisory and severity. A target Trivy couldn't scan is a **scan
error** in the issue. If the script itself fails, the workflow's last step comments on or opens the
issue, so a broken scan isn't silent. A body over GitHub's size limit is cut to fewer rows per
target; the run's summary page always has every row.

## Running it by hand

Needs git, Docker and python3. Trivy runs from the image the script pins (`TRIVY_IMAGE`).

```bash
scripts/security_scan.py --list                     # what it would scan
scripts/security_scan.py --dry-run                  # scan both, print the issue body
scripts/security_scan.py --dry-run --only uv.lock   # just the lockfile
gh workflow run security-scan.yaml --repo drewlindsey/speaches   # the workflow itself
```

`--dry-run` never touches GitHub. `SECURITY_SCAN_CACHE=~/.cache/security-scan` keeps Trivy's
database (about 1.6 GB) between runs. Without `--dry-run` the script updates the issue on
`drewlindsey/speaches` only (`REPO` in the script), never upstream.

## Acting on a finding

Bump the package in `pyproject.toml`, `uv lock`, and merge: `fork-image.yaml` publishes a new
`master-cpu`, and the next scan reads it. An OS package in the image clears when the base image
(`ubuntu:24.04`) is rebuilt upstream and this fork's image is rebuilt on it.

To bump Trivy, take the newest release at least 7 days old and resolve its index digest
(`docker buildx imagetools inspect aquasec/trivy:<version>`).

Scheduled workflows in a public repo stop after 60 days without activity there; if the Actions tab
shows **security-scan** disabled, enable it there.
