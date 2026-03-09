"""Tests for openmic.version module."""

import json
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from openmic.version import (
    CACHE_TTL,
    detect_install_method,
    get_latest_version,
    get_upgrade_command,
    get_version,
)


def test_get_version():
    """get_version returns the installed package version."""
    with patch("openmic.version.version", return_value="0.1.0"):
        assert get_version() == "0.1.0"


def test_get_version_not_installed():
    """get_version returns fallback when package is not installed."""
    from importlib.metadata import PackageNotFoundError

    with patch("openmic.version.version", side_effect=PackageNotFoundError):
        assert get_version() == "0.0.0-dev"


def test_get_latest_version_cached():
    """Returns cached version without HTTP call when cache is fresh."""
    config = {
        "update_latest_version": "0.2.0",
        "update_checked_at": time.time(),
    }
    with patch("openmic.version.urlopen") as mock_urlopen:
        result = get_latest_version(config)
        assert result == "0.2.0"
        mock_urlopen.assert_not_called()


def test_get_latest_version_stale_cache():
    """Fetches from GitHub when cache is stale."""
    config = {
        "update_latest_version": "0.1.0",
        "update_checked_at": time.time() - CACHE_TTL - 100,
    }
    response = BytesIO(json.dumps({"tag_name": "v0.3.0"}).encode())
    with patch("openmic.version.urlopen", return_value=response):
        result = get_latest_version(config)
        assert result == "0.3.0"
        assert config["update_latest_version"] == "0.3.0"


def test_get_latest_version_fresh():
    """Fetches from GitHub when no cache exists."""
    config = {}
    response = BytesIO(json.dumps({"tag_name": "v0.2.0"}).encode())
    with patch("openmic.version.urlopen", return_value=response):
        result = get_latest_version(config)
        assert result == "0.2.0"
        assert config["update_latest_version"] == "0.2.0"
        assert "update_checked_at" in config


def test_get_latest_version_no_internet():
    """Returns None when GitHub is unreachable."""
    from urllib.error import URLError

    config = {}
    with patch("openmic.version.urlopen", side_effect=URLError("no internet")):
        result = get_latest_version(config)
        assert result is None


def test_get_latest_version_bad_response():
    """Returns None on invalid JSON response."""
    config = {}
    response = BytesIO(b"not json")
    with patch("openmic.version.urlopen", return_value=response):
        result = get_latest_version(config)
        assert result is None


def test_detect_install_method_editable():
    """Detects editable install via direct_url.json."""
    mock_dist = MagicMock()
    mock_dist.read_text.return_value = json.dumps({
        "dir_info": {"editable": True},
    })
    with patch("importlib.metadata.distribution", return_value=mock_dist):
        assert detect_install_method() == "editable"


def test_detect_install_method_pipx():
    """Detects pipx install."""
    mock_dist = MagicMock()
    mock_dist.read_text.return_value = None  # No direct_url.json

    with (
        patch("importlib.metadata.distribution", return_value=mock_dist),
        patch("shutil.which", side_effect=lambda x: "/usr/bin/pipx" if x == "pipx" else None),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout="openmic 0.1.0\n", returncode=0)
        assert detect_install_method() == "pipx"


def test_detect_install_method_pip_fallback():
    """Falls back to pip when pipx and uv are not available."""
    mock_dist = MagicMock()
    mock_dist.read_text.return_value = None

    with (
        patch("importlib.metadata.distribution", return_value=mock_dist),
        patch("shutil.which", return_value=None),
    ):
        assert detect_install_method() == "pip"


def test_get_upgrade_command_all_methods():
    """Verify command lists for all install methods."""
    assert get_upgrade_command("pipx") == ["pipx", "upgrade", "openmic"]
    assert get_upgrade_command("uv") == ["uv", "tool", "upgrade", "openmic"]

    pip_cmd = get_upgrade_command("pip")
    assert pip_cmd is not None
    assert "-m" in pip_cmd
    assert "pip" in pip_cmd
    assert "--upgrade" in pip_cmd

    assert get_upgrade_command("editable") is None
    assert get_upgrade_command("unknown") is None


def test_main_version_flag(capsys):
    """openmic --version prints version to stdout."""
    with patch("sys.argv", ["openmic", "--version"]):
        from openmic.app import main

        with patch("openmic.version.version", return_value="0.1.0"):
            main()
        output = capsys.readouterr().out
        assert "openmic 0.1.0" in output
