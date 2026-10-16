import pytest


@pytest.fixture(autouse=True)
def _clean_assets():
    """Overrides the unit-test cleanup: integration tests share assets within the module."""
    yield
