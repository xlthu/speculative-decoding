import torch

from .base import Base
from .utils import chain_attention_mask, chain_position_ids
from .cache import DynamicCache

__all__ = ["AutoRegressive"]


class AutoRegressive(Base):

    ### Input

    def get_input(
        self, all_tokens: torch.Tensor, cache: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_past = cache.get_seq_length()

        in_tokens = all_tokens[:, n_past:]
        n_token = in_tokens.shape[1]  # [1, n_token]

        attention_mask = chain_attention_mask(n_past, n_token, self.dtype, self.device)
        position_ids = chain_position_ids(n_past, n_token, self.device)

        return in_tokens, attention_mask, position_ids

    ### Output

    def obtain_output(
        self,
        in_tokens: torch.Tensor,
        logits: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        # No need to update cache
        return torch.argmax(logits[:, -1:], dim=-1)  # [1, 1]
