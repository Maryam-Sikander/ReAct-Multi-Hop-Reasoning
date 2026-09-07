import os
import time
from google import genai
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()

class LLMResponse:
    def __init__(self, text, prompt_tokens, completion_tokens):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class LLMClient:
    def __init__(self, model_id, temperature=0.0, max_output_tokens=512,
                 retry_attempts=4, retry_backoff_s=5):
        self.model_id = model_id
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.retry_attempts = retry_attempts
        self.retry_backoff_s = retry_backoff_s

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in your .env")
        self.client = genai.Client(api_key=api_key)

    def complete(self, prompt, stop=None):
        config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if stop:
            config["stop_sequences"] = stop

        last_err = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                # Use standard Google GenAI SDK generation method
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=str(prompt),
                    config=config,
                )
                
                # Extract response text safely
                text = getattr(response, "text", None) or str(response)

                # Token count parsing
                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(usage, "prompt_token_count", 0)
                completion_tokens = getattr(usage, "candidates_token_count", 0)

                return LLMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            except errors.APIError as e:
                last_err = e
                # Wait longer if hitting rate limits (HTTP 429)
                sleep_time = self.retry_backoff_s * (2 ** (attempt - 1))
                print(f"API Error ({e.code}). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            except Exception as e:
                last_err = e
                time.sleep(self.retry_backoff_s * attempt)

        raise RuntimeError(f"Gemini call failed after {self.retry_attempts} tries: {last_err}")