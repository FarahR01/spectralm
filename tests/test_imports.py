"""Unit tests for the spectralm package."""

import pytest


class TestPackageImport:
    """Test basic package imports."""

    def test_import_spectralm(self):
        """Test that spectralm can be imported."""
        import spectralm

        assert hasattr(spectralm, "__version__")

    def test_import_models(self):
        """Test that models can be imported."""
        from spectralm.models import SpectraLM, SpectraLMConfig

        assert SpectraLM is not None
        assert SpectraLMConfig is not None

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        import spectralm

        version = spectralm.__version__
        parts = version.split(".")
        assert len(parts) >= 3
        for part in parts:
            assert part.isdigit()
