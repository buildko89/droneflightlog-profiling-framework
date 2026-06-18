import json
import os
import google.generativeai as genai
from openai import OpenAI
import anthropic


SUPPORTED_LLM_TYPES = ("gemini", "openai", "anthropic", "dummy")
DEFAULT_LLM_CONFIG_PATH = "llm_config.json"
SUPPORTED_LLM_MODELS = {
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    "dummy": [
        "dummy-model",
    ],
}


def get_model_choices(service, configured_model=None):
    """
    Returns UI model choices for a service, preserving custom configured models.
    """
    choices = list(SUPPORTED_LLM_MODELS.get(service, []))
    if configured_model and configured_model not in choices:
        choices.insert(0, configured_model)
    return choices


def load_llm_config(config_path=DEFAULT_LLM_CONFIG_PATH, required=False):
    """
    Loads LLM service/model settings from JSON.

    Supported formats:
    {
      "service": "gemini",
      "model": "gemini-2.5-flash"
    }

    or:
    {
      "llm": {
        "service": "gemini",
        "model": "gemini-2.5-flash"
      }
    }
    """
    if not config_path:
        return {}

    if not os.path.exists(config_path):
        if required:
            raise FileNotFoundError(f"LLM config file not found: {config_path}")
        return {}

    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("LLM config must be a JSON object.")

    llm_config = config.get("llm", config)
    if not isinstance(llm_config, dict):
        raise ValueError("LLM config field 'llm' must be a JSON object.")

    service = llm_config.get("service", llm_config.get("llm"))
    model = llm_config.get("model", llm_config.get("model_name"))
    mode = llm_config.get("mode")

    resolved = {}
    if service is not None:
        service = str(service).strip().lower()
        if service not in SUPPORTED_LLM_TYPES:
            raise ValueError(
                f"Unsupported LLM service '{service}'. "
                f"Choose one of: {', '.join(SUPPORTED_LLM_TYPES)}"
            )
        resolved["service"] = service

    if model is not None and str(model).strip():
        resolved["model"] = str(model).strip()

    if mode is not None:
        resolved["mode"] = str(mode).strip().lower()

    return resolved


def resolve_llm_settings(
    service=None,
    model_name=None,
    config_path=DEFAULT_LLM_CONFIG_PATH,
    mode=None,
):
    """
    Resolves LLM settings with this precedence:
    explicit arguments > JSON config > built-in defaults.
    """
    config = load_llm_config(config_path)
    config_service = config.get("service")
    resolved_service = service or config_service or "gemini"

    if model_name:
        resolved_model = model_name
    elif service and config_service and service != config_service:
        resolved_model = None
    else:
        resolved_model = config.get("model")

    # Resolve mode: CLI arg > JSON config > default 'api'
    resolved_mode = mode or config.get("mode") or "api"
    resolved_mode = resolved_mode.lower()
    if resolved_mode not in ("api", "export"):
        resolved_mode = "api"

    if resolved_service not in SUPPORTED_LLM_TYPES:
        raise ValueError(
            f"Unsupported LLM service '{resolved_service}'. "
            f"Choose one of: {', '.join(SUPPORTED_LLM_TYPES)}"
        )

    return {
        "service": resolved_service,
        "model": resolved_model,
        "mode": resolved_mode,
    }


def get_required_api_key_name(service):
    required_keys = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return required_keys.get(service)


def create_llm_client(
    service=None,
    model_name=None,
    config_path=DEFAULT_LLM_CONFIG_PATH,
    require_api_key=True,
):
    settings = resolve_llm_settings(
        service=service,
        model_name=model_name,
        config_path=config_path,
    )
    client_kwargs = {}
    if settings["model"]:
        client_kwargs["model_name"] = settings["model"]
    client_kwargs["require_api_key"] = require_api_key

    if settings["service"] == "gemini":
        return GeminiClient(**client_kwargs)
    if settings["service"] == "openai":
        return OpenAIClient(**client_kwargs)
    if settings["service"] == "anthropic":
        return AnthropicClient(**client_kwargs)
    if settings["service"] == "dummy":
        return DummyClient(**client_kwargs)

    raise ValueError(f"Unsupported LLM service: {settings['service']}")


class BaseLLMClient:
    """
    Abstract base class for LLM clients.
    """
    def generate_text(self, prompt: str) -> str:
        """
        Generates text based on the provided prompt.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_text method.")

class GeminiClient(BaseLLMClient):
    """
    Client for Google Gemini API.
    """
    def __init__(self, model_name="gemini-2.5-flash", require_api_key=True):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if require_api_key and not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # Ensure model name starts with 'models/'
        self.target_model = self.model_name
        if not self.target_model.startswith("models/"):
            self.target_model = f"models/{self.target_model}"

    def generate_text(self, prompt: str) -> str:
        """
        Calls Gemini API and returns the generated text.
        """
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment variables.")
        try:
            model = genai.GenerativeModel(self.target_model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI API.
    """
    def __init__(self, model_name="gpt-4o", require_api_key=True):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if require_api_key and not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate_text(self, prompt: str) -> str:
        """
        Calls OpenAI API and returns the generated text.
        """
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY not found in environment variables.")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

class AnthropicClient(BaseLLMClient):
    """
    Client for Anthropic (Claude) API.
    """
    def __init__(self, model_name="claude-3-5-sonnet-20240620", require_api_key=True):
        self.model_name = model_name
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if require_api_key and not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    def generate_text(self, prompt: str) -> str:
        """
        Calls Anthropic API and returns the generated text.
        """
        if self.client is None:
            raise RuntimeError("ANTHROPIC_API_KEY not found in environment variables.")
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

class DummyClient(BaseLLMClient):
    """
    Dummy client for testing purposes.
    Does not require any API keys.
    """
    def __init__(self, model_name="dummy-model", require_api_key=True):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        """
        Returns a fixed dummy response.
        """
        return f"これはダミーLLM({self.model_name})の回答です。プロンプトの長さは {len(prompt)} 文字でした。"
