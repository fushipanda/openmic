"""OpenMic setup wizard — interactive onboarding for first-time users."""

import os
import subprocess
import sys

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, OptionList, Static
from textual.widgets.option_list import Option
from textual.binding import Binding
from textual.worker import Worker
from textual import work

from openmic.app import (
    _load_config,
    _save_config,
    _update_env_file,
    MODEL_REGISTRY,
    CONFIG_DIR,
    BANNER,
    OPENMIC_THEME,
)


# Provider → pip packages to install
PROVIDER_DEPS: dict[str, list[str]] = {
    "anthropic": ["langchain-anthropic>=0.1.0", "langchain-openai>=0.0.5"],
    "openai": ["langchain-openai>=0.0.5"],
    "gemini": ["langchain-google-genai>=2.0.0", "langchain-openai>=0.0.5"],
    "openrouter": ["langchain-openai>=0.0.5"],
}


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


class SetupScreen(Screen):
    """Multi-step setup wizard screen."""

    DEFAULT_CSS = """
    SetupScreen {
        align: center middle;
    }

    #setup-container {
        width: 64;
        max-height: 32;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }

    #setup-banner {
        text-align: center;
        color: $accent;
        margin-bottom: 1;
    }

    #setup-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #setup-body {
        margin-bottom: 1;
    }

    #setup-providers {
        height: auto;
        max-height: 8;
        margin-bottom: 1;
    }

    #setup-input {
        margin-bottom: 1;
    }

    #setup-status {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    #setup-nav {
        height: 3;
        align: center middle;
    }

    #setup-nav Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._step = 0  # 0=welcome, 1=provider, 2=installing, 3=api_keys, 4=done
        self._provider = ""
        self._required_keys: list[tuple[str, str]] = []
        self._current_key_index = 0
        self._collected_keys: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-container"):
            yield Static(BANNER, id="setup-banner")
            yield Static("", id="setup-title")
            yield Static("", id="setup-body")
            yield OptionList(id="setup-providers")
            yield Input(id="setup-input", password=True)
            yield Static("", id="setup-status")
            from textual.containers import Horizontal
            with Horizontal(id="setup-nav"):
                yield Button("Continue", id="btn-continue", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self._show_welcome()

    def _show_welcome(self) -> None:
        self._step = 0
        self.query_one("#setup-title", Static).update("Let's get you set up")
        self.query_one("#setup-body", Static).update(
            "This wizard will configure your LLM provider,\n"
            "install dependencies, and set up API keys."
        )
        self.query_one("#setup-providers", OptionList).display = False
        self.query_one("#setup-input", Input).display = False
        self.query_one("#setup-status", Static).update("")
        self.query_one("#btn-continue", Button).label = "Get Started"

    def _show_provider_select(self) -> None:
        self._step = 1
        self.query_one("#setup-title", Static).update("Choose your LLM provider")
        self.query_one("#setup-body", Static).update(
            "This controls which AI model answers your queries\n"
            "and generates meeting notes."
        )

        option_list = self.query_one("#setup-providers", OptionList)
        option_list.clear_options()
        option_list.add_option(Option("Anthropic (Claude) — recommended", id="anthropic"))
        option_list.add_option(Option("OpenAI (GPT)", id="openai"))
        option_list.add_option(Option("Google (Gemini)", id="gemini"))
        option_list.add_option(Option("OpenRouter", id="openrouter"))
        option_list.display = True
        option_list.highlighted = 0

        self.query_one("#setup-input", Input).display = False
        self.query_one("#setup-status", Static).update("")
        self.query_one("#btn-continue", Button).label = "Continue"

    def _show_installing(self) -> None:
        self._step = 2
        label = MODEL_REGISTRY.get(self._provider, {}).get("label", self._provider)
        self.query_one("#setup-title", Static).update("Installing dependencies")
        self.query_one("#setup-body", Static).update(
            f"Installing packages for {label}..."
        )
        self.query_one("#setup-providers", OptionList).display = False
        self.query_one("#setup-input", Input).display = False
        self.query_one("#setup-status", Static).update("This may take a moment...")
        self.query_one("#btn-continue", Button).display = False
        self._install_provider_deps()

    @work(thread=True)
    def _install_provider_deps(self) -> None:
        """Install provider dependencies in a worker thread."""
        deps = PROVIDER_DEPS.get(self._provider, [])
        if not deps:
            self.app.call_from_thread(self._on_install_complete, True, "")
            return

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", *deps],
                capture_output=True,
                text=True,
                timeout=120,
            )
            success = result.returncode == 0
            error = result.stderr if not success else ""
            self.app.call_from_thread(self._on_install_complete, success, error)
        except Exception as e:
            self.app.call_from_thread(self._on_install_complete, False, str(e))

    def _on_install_complete(self, success: bool, error: str) -> None:
        if success:
            self.query_one("#setup-status", Static).update("Dependencies installed!")
            self._show_api_keys()
        else:
            self.query_one("#setup-status", Static).update(
                f"Install failed: {error[:100]}\nYou can install manually later."
            )
            self.query_one("#btn-continue", Button).display = True
            self.query_one("#btn-continue", Button).label = "Continue Anyway"
            # Allow continuing even if install fails — user can fix later
            self._step = 2  # stay on this step, continue goes to api keys

    def _show_api_keys(self) -> None:
        self._step = 3
        self._required_keys = _get_required_keys(self._provider)
        self._current_key_index = 0
        self._collected_keys = {}
        self._show_next_key()

    def _show_next_key(self) -> None:
        # Skip keys that are already set in the environment
        while self._current_key_index < len(self._required_keys):
            env_key, label = self._required_keys[self._current_key_index]
            if os.environ.get(env_key):
                self._collected_keys[env_key] = os.environ[env_key]
                self._current_key_index += 1
                continue
            break

        if self._current_key_index >= len(self._required_keys):
            self._show_done()
            return

        env_key, label = self._required_keys[self._current_key_index]
        remaining = len(self._required_keys) - self._current_key_index
        self.query_one("#setup-title", Static).update(
            f"API Keys ({self._current_key_index + 1}/{len(self._required_keys)})"
        )
        self.query_one("#setup-body", Static).update(f"Enter your {label}:")
        self.query_one("#setup-providers", OptionList).display = False

        input_widget = self.query_one("#setup-input", Input)
        input_widget.value = ""
        input_widget.placeholder = env_key
        input_widget.display = True
        input_widget.focus()

        self.query_one("#setup-status", Static).update("")
        self.query_one("#btn-continue", Button).display = True
        self.query_one("#btn-continue", Button).label = "Next" if remaining > 1 else "Finish"

    def _show_done(self) -> None:
        self._step = 4
        self.query_one("#setup-title", Static).update("You're all set!")
        self.query_one("#setup-body", Static).update(
            "Configuration saved. Run `openmic` to start."
        )
        self.query_one("#setup-providers", OptionList).display = False
        self.query_one("#setup-input", Input).display = False

        # Save everything
        config = _load_config()
        config["setup_complete"] = True
        config["llm_provider"] = self._provider
        # Pick first model for the provider as default
        models = MODEL_REGISTRY.get(self._provider, {}).get("models", [])
        if models:
            config["llm_model"] = models[0][0]
        _save_config(config)

        # Save API keys to env and .env file
        for env_key, value in self._collected_keys.items():
            os.environ[env_key] = value
            _update_env_file(env_key, value)

        # Save provider to .env
        os.environ["LLM_PROVIDER"] = self._provider
        _update_env_file("LLM_PROVIDER", self._provider)

        skipped = sum(
            1 for k, _ in self._required_keys if k not in self._collected_keys
        )
        status_parts = ["Configuration saved."]
        if skipped:
            status_parts.append(f"{skipped} key(s) already configured.")
        self.query_one("#setup-status", Static).update(" ".join(status_parts))
        self.query_one("#btn-continue", Button).label = "Done"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.app.exit()
            return

        if event.button.id == "btn-continue":
            if self._step == 0:
                self._show_provider_select()
            elif self._step == 1:
                option_list = self.query_one("#setup-providers", OptionList)
                if option_list.highlighted is not None:
                    option = option_list.get_option_at_index(option_list.highlighted)
                    self._provider = str(option.id)
                    self._show_installing()
            elif self._step == 2:
                # Continue after failed install
                self._show_api_keys()
            elif self._step == 3:
                self._submit_current_key()
            elif self._step == 4:
                self.app.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._step == 3:
            self._submit_current_key()

    def _submit_current_key(self) -> None:
        input_widget = self.query_one("#setup-input", Input)
        value = input_widget.value.strip()
        if not value:
            self.query_one("#setup-status", Static).update("API key cannot be empty.")
            return

        env_key, _ = self._required_keys[self._current_key_index]
        self._collected_keys[env_key] = value
        self._current_key_index += 1
        self._show_next_key()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._step == 1:
            self._provider = str(event.option.id)
            self._show_installing()

    def action_cancel(self) -> None:
        self.app.exit()


class SetupApp(App):
    """Minimal Textual app for the setup wizard."""

    CSS = """
    Screen {
        background: $background;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(OPENMIC_THEME)
        self.theme = "openmic"

    def on_mount(self) -> None:
        self.push_screen(SetupScreen())


def run_setup() -> None:
    """Entry point for the setup wizard."""
    app = SetupApp()
    app.run()


if __name__ == "__main__":
    run_setup()
