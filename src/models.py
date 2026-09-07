"""Local Hugging Face model client used by both experiments."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMResponse:
    def __init__(self, text, prompt_tokens, completion_tokens):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class HFLocalClient:
    def __init__(self, model_id, max_new_tokens=200, temperature=0.0):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant_config, device_map="auto",
        )

    def complete(self, prompt, stop=None, max_new_tokens=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        budget = max_new_tokens or self.max_new_tokens

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 0.01),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion_tokens = output_ids.shape[1] - prompt_tokens
        text = self.tokenizer.decode(
            output_ids[0][prompt_tokens:], skip_special_tokens=True
        )

        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
