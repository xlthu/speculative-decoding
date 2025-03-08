import torch

from .base import Base

__all__ = ["HF"]


class HF(Base):
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs):
        attention_mask = torch.ones_like(input_ids)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            # Greedy decoding
            do_sample=False,
            top_k=None,
            top_p=None,
            temperature=None,
            repetition_penalty=None,
        )
