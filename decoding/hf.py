import torch

from .base import Base
from .stat import Stat
from .cache import DynamicCache

__all__ = ["HF"]


class HF(Base):
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        cache: DynamicCache = None,
        stat: Stat = None,
    ):
        assert (
            cache is None and past_ids is None
        ), "Multi-turn conversations are not supported"
        assert stat is None, "Statistic is not supported"

        attention_mask = torch.ones_like(input_ids)
        output_ids = self.model.generate(
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

        return {"output_ids": output_ids, "stat": stat, "cache": None}
