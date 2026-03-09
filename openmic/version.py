"""Version display, update check, and self-update logic."""

import json
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


GITHUB_RELEASES_URL = "https://api.github.com/repos/fushipanda/openmic/releases/latest"
CACHE_TTL = 86400  # 24 hours


def get_version() -> str:
    """Return the installed version of openmic."""
    try:
        return version("openmic")
    except PackageNotFoundError:
        return "0.0.0-dev"


def get_latest_version(config: dict) -> str | None:
    """Check GitHub for the latest release version, with 24h caching.

    Updates config dict in-place with cache values. Returns None on error.
    """
    cached = config.get("update_latest_version")
    checked_at = config.get("update_checked_at", 0)

    if cached and (time.time() - checked_at) < CACHE_TTL:
        return cached

    try:
        req = urlopen(GITHUB_RELEASES_URL, timeout=5)  # noqa: S310
        data = json.loads(req.read().decode())
        tag = data.get("tag_name", "")
        latest = tag.lstrip("v")
        if latest:
            config["update_latest_version"] = latest
            config["update_checked_at"] = time.time()
            return latest
    except (URLError, OSError, json.JSONDecodeError, KeyError):
        pass

    return None


def detect_install_method() -> str:
    """Detect how openmic was installed: pipx, uv, pip, editable, or unknown."""
    # Check for editable install via direct_url.json
    try:
        from importlib.metadata import packages_distributions  # noqa: F401

        dist_files = version  # just to access metadata
        import importlib.metadata as md

        dist = md.distribution("openmic")
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            url_data = json.loads(direct_url)
            if url_data.get("dir_info", {}).get("editable", False):
                return "editable"
    except (PackageNotFoundError, TypeError, json.JSONDecodeError, FileNotFoundError):
        pass

    # Check pipx
    if shutil.which("pipx"):
        try:
            result = subprocess.run(
                ["pipx", "list", "--short"],
                capture_output=True, text=True, timeout=10,
            )
            if "openmic" in result.stdout:
                return "pipx"
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Check uv
    if shutil.which("uv"):
        try:
            result = subprocess.run(
                ["uv", "tool", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if "openmic" in result.stdout:
                return "uv"
        except (subprocess.TimeoutExpired, OSError):
            pass

    return "pip"


def get_upgrade_command(method: str) -> list[str] | None:
    """Return the command list to upgrade openmic for the given install method."""
    commands = {
        "pipx": ["pipx", "upgrade", "openmic"],
        "uv": ["uv", "tool", "upgrade", "openmic"],
        "pip": [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "git+https://github.com/fushipanda/openmic.git",
        ],
    }
    return commands.get(method)


def run_update() -> None:
    """Entry point for `openmic update` subcommand."""
    current = get_version()
    print(f"openmic v{current}")

    method = detect_install_method()
    print(f"Install method: {method}")

    if method == "editable":
        print("Editable install detected. Run `git pull` to update.")
        return

    cmd = get_upgrade_command(method)
    if cmd is None:
        print("Could not determine upgrade command.")
        return

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("Update complete!")
    else:
        print(f"Update failed (exit code {result.returncode}).")
