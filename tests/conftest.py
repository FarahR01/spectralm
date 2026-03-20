"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture(scope="session")
def setup_session():
    """Set up test session."""
    yield


@pytest.fixture
def sample_fixture():
    """Sample fixture for tests."""
    return {"test": "data"}
