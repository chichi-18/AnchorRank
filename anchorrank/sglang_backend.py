from __future__ import annotations

from typing import Sequence

from .types import TextGenerator


class SGLangGenerator(TextGenerator):
    """Lazy SGLang backend, keeping package import usable without SGLang installed."""

    def __init__(self, model_path: str, context_length: int = 12288, tp_size: int = 1, dp_size: int = 1, seed: int = 42, enable_thinking: bool = False):
        try:
            import sglang as sgl
        except ImportError as error:
            raise ImportError("Install the optional 'sglang' dependency to use SGLangGenerator.") from error
        self.engine = sgl.Engine(model_path=model_path, context_length=context_length, tp_size=tp_size, dp_size=dp_size, random_seed=seed)
        self.tokenizer = self.engine.tokenizer_manager.tokenizer
        self.enable_thinking = enable_thinking

    def generate(self, prompts: Sequence[str], *, max_new_tokens: int) -> list[str]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        outputs = self.engine.generate(inputs, {"temperature": 0.0, "max_new_tokens": max_new_tokens, "skip_special_tokens": True})
        return [output["text"] for output in outputs]

    def shutdown(self) -> None:
        self.engine.shutdown()
