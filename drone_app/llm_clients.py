import os
import google.generativeai as genai
from openai import OpenAI
import anthropic

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
