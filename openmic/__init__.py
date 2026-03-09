"""OpenMic - CLI/TUI meeting transcription tool."""


def __getattr__(name):
    if name == "__version__":
        from importlib.metadata import version
        return version("openmic")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
