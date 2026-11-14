from httpx import ASGITransport, AsyncClient
import pytest
from pytest_mock import MockerFixture

from speaches.executors.shared.errors import UnsupportedResponseFormatError
from speaches.main import create_app
from tests.conftest import DEFAULT_CONFIG


@pytest.mark.asyncio
async def test_unsupported_response_format_error_is_a_400(mocker: MockerFixture) -> None:
    mocker.patch("speaches.main.get_config", return_value=DEFAULT_CONFIG)
    app = create_app()

    def unsupported() -> None:
        raise UnsupportedResponseFormatError("srt", "some/model")

    app.add_api_route("/unsupported", unsupported)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/unsupported")

    assert response.status_code == 400
    assert response.json() == {"detail": "'srt' response format is not supported for 'some/model' model."}
