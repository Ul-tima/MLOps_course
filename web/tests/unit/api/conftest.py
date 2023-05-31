import main
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(main.app)
