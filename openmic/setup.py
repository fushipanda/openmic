"""OpenMic setup wizard — interactive CLI onboarding for first-time users."""

import getpass
import os
import subprocess
import sys

from openmic.app import (
    _load_config,
    _save_config,
    _update_env_file,
    MODEL_REGISTRY,
    BANNER,
)


# Provider → pip packages to install
PROVIDER_DEPS: dict[str, list[str]] = {
    "anthropic": ["langchain-anthropic>=0.1.0", "langchain-openai>=0.0.5"],
    "openai": ["langchain-openai>=0.0.5"],
    "gemini": ["langchain-google-genai>=2.0.0", "langchain-openai>=0.0.5"],
    "openrouter": ["langchain-openai>=0.0.5"],
}

KEY_URLS = {
    "ELEVENLABS_API_KEY": "https://elevenlabs.io/app/settings/api-keys",
    "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
    "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
    "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
    "OPENROUTER_API_KEY": "https://openrouter.ai/settings/keys",
}

PROVIDERS = [
    ("anthropic", "Anthropic (Claude) — recommended"),
    ("openai", "OpenAI (GPT)"),
    ("gemini", "Google (Gemini)"),
    ("openrouter", "OpenRouter"),
]


def _get_required_keys(provider: str) -> list[tuple[str, str]]:
    """Return list of (env_key, label) tuples required for the given provider."""
    keys = [
        ("ELEVENLABS_API_KEY", "ElevenLabs API key (transcription)"),
        ("OPENAI_API_KEY", "OpenAI API key (embeddings)"),
    ]
    if provider == "anthropic":
        keys.append(("ANTHROPIC_API_KEY", "Anthropic API key (Claude)"))
    elif provider == "gemini":
        keys.append(("GEMINI_API_KEY", "Google Gemini API key"))
    elif provider == "openrouter":
        keys.append(("OPENROUTER_API_KEY", "OpenRouter API key"))
    # openai: no extra key needed (already have OPENAI_API_KEY)
    return keys


def _print_banner() -> None:
    """Print the OpenMic banner."""
    print(BANNER)
    print()


def _prompt_provider() -> str:
    """Prompt user to choose an LLM provider. Returns provider key."""
    print("Choose your LLM provider:")
    for i, (_, label) in enumerate(PROVIDERS, 1):
        print(f"  {i}) {label}")
    print()

    while True:
        choice = input("Enter choice [1]: ").strip()
        if choice == "":
            return PROVIDERS[0][0]
        try:
            idx = int(choice)
            if 1 <= idx <= len(PROVIDERS):
                return PROVIDERS[idx - 1][0]
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(PROVIDERS)}.")


def _install_deps(provider: str) -> None:
    """Install pip dependencies for the chosen provider."""
    deps = PROVIDER_DEPS.get(provider, [])
    if not deps:
        return

    label = MODEL_REGISTRY.get(provider, {}).get("label", provider)
    print(f"\nInstalling dependencies for {label}...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *deps],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print("Done!")
        else:
            print(f"Warning: install failed — {result.stderr[:200]}")
            print("You can install manually later.")
    except Exception as e:
        print(f"Warning: install failed — {e}")
        print("You can install manually later.")


def _prompt_api_keys(provider: str) -> dict[str, str]:
    """Prompt for API keys, skipping ones already in env. Returns collected keys."""
    required = _get_required_keys(provider)
    collected: dict[str, str] = {}
    already_set = 0

    for i, (env_key, label) in enumerate(required, 1):
        print(f"\nAPI Keys ({i}/{len(required)})")

        if os.environ.get(env_key):
            print(f"  {label}")
            print("  [already set, skipping]")
            collected[env_key] = os.environ[env_key]
            already_set += 1
            continue

        print(f"  Enter your {label}")
        url = KEY_URLS.get(env_key)
        if url:
            print(f"  Get your key at: {url}")

        while True:
            value = getpass.getpass("  Key: ").strip()
            if value:
                collected[env_key] = value
                break
            print("  API key cannot be empty.")

    return collected


def _save_setup(provider: str, collected_keys: dict[str, str]) -> None:
    """Save provider config and API keys to settings.json and .env."""
    config = _load_config()
    config["setup_complete"] = True
    config["llm_provider"] = provider

    # Pick first model for the provider as default
    models = MODEL_REGISTRY.get(provider, {}).get("models", [])
    if models:
        config["llm_model"] = models[0][0]
    _save_config(config)

    # Save API keys to env and .env file
    for env_key, value in collected_keys.items():
        os.environ[env_key] = value
        _update_env_file(env_key, value)

    # Save provider to .env
    os.environ["LLM_PROVIDER"] = provider
    _update_env_file("LLM_PROVIDER", provider)


def run_setup() -> None:
    """Run the CLI setup wizard."""
    try:
        _print_banner()
        print("Welcome to OpenMic setup!")
        print("This will configure your LLM provider, install")
        print("dependencies, and set up API keys.")
        print()
        input("Press Enter to continue (or Ctrl+C to cancel)... ")

        provider = _prompt_provider()
        _install_deps(provider)
        collected_keys = _prompt_api_keys(provider)
        _save_setup(provider, collected_keys)

        # Summary
        label = MODEL_REGISTRY.get(provider, {}).get("label", provider)
        models = MODEL_REGISTRY.get(provider, {}).get("models", [])
        model_name = models[0][0] if models else "default"

        required = _get_required_keys(provider)
        already_set = sum(1 for k, _ in required if os.environ.get(k) and k not in collected_keys)

        print("\nSetup complete!")
        print(f"  Provider: {label}")
        print(f"  Model:    {model_name}")
        total = len(collected_keys)
        if already_set:
            print(f"  Keys configured: {total} ({already_set} already set)")
        else:
            print(f"  Keys configured: {total}")
        print("Run `openmic` to start.")

    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")


if __name__ == "__main__":
    run_setup()
