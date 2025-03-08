import torch
from transformers import DynamicCache

from .base import Base

__all__ = ["AutoRegressive"]


class AutoRegressive(Base):

    ### Input

    def get_input(
        self, all_tokens: torch.Tensor, cache: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_past = cache.get_seq_length()

        in_tokens = all_tokens[:, n_past:]
        n_token = in_tokens.shape[1]  # [1, n_token]

        attention_mask = self.prepare_attention_mask(n_past, n_token)
        position_ids = self.prepare_position_ids(n_past, n_token)

        return in_tokens, attention_mask, position_ids

    def prepare_attention_mask(self, n_past: int, n_token: int):
        # return attention_mask: [1, 1, n_token, n_past + n_token]
        # 0: allow attention, -inf: not allow attention
        min_dtype = torch.finfo(self.dtype).min
        mask = torch.full(
            (n_token, n_past + n_token),
            fill_value=min_dtype,
            dtype=self.dtype,
            device=self.device,
        )
        mask = torch.triu(mask, diagonal=n_past + 1)
        mask = mask.reshape(1, 1, *mask.shape)

        return mask

    def prepare_position_ids(self, n_past: int, n_token: int):
        # return position_ids: [1, n_token]
        return torch.arange(n_past, n_token + n_past, device=self.device).unsqueeze(0)

    ### Output

    def obtain_output(
        self,
        in_tokens: torch.Tensor,
        logits: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        # No need to update cache
        return torch.argmax(logits[:, -1:], dim=-1)  # [1, 1]
