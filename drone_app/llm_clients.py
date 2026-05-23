import json
import os
import google.generativeai as genai
from openai import OpenAI
import anthropic


SUPPORTED_LLM_TYPES = ("gemini", "openai", "anthropic", "dummy")
DEFAULT_LLM_CONFIG_PATH = "llm_config.json"


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

    return resolved


def resolve_llm_settings(
    service=None,
    model_name=None,
    config_path=DEFAULT_LLM_CONFIG_PATH,
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

    if resolved_service not in SUPPORTED_LLM_TYPES:
        raise ValueError(
            f"Unsupported LLM service '{resolved_service}'. "
            f"Choose one of: {', '.join(SUPPORTED_LLM_TYPES)}"
        )

    return {
        "service": resolved_service,
        "model": resolved_model,
    }


def get_required_api_key_name(service):
    required_keys = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return required_keys.get(service)


def create_llm_client(service=None, model_name=None, config_path=DEFAULT_LLM_CONFIG_PATH):
    settings = resolve_llm_settings(
        service=service,
        model_name=model_name,
        config_path=config_path,
    )
    client_kwargs = {}
    if settings["model"]:
        client_kwargs["model_name"] = settings["model"]

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
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        genai.configure(api_key=self.api_key)
        
        # Ensure model name starts with 'models/'
        self.target_model = self.model_name
        if not self.target_model.startswith("models/"):
            self.target_model = f"models/{self.target_model}"

    def generate_text(self, prompt: str) -> str:
        """
        Calls Gemini API and returns the generated text.
        """
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
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(api_key=self.api_key)

    def generate_text(self, prompt: str) -> str:
        """
        Calls OpenAI API and returns the generated text.
        """
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
    def __init__(self, model_name="claude-3-5-sonnet-20240620"):
        self.model_name = model_name
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_text(self, prompt: str) -> str:
        """
        Calls Anthropic API and returns the generated text.
        """
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
    def __init__(self, model_name="dummy-model"):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        """
        Returns a fixed dummy response.
        """
        return f"これはダミーLLM({self.model_name})の回答です。プロンプトの長さは {len(prompt)} 文字でした。"
