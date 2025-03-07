import torch
from transformers import DynamicCache

from .base import Base

__all__ = ["AutoRegressive"]


class AutoRegressive(Base):
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs):
        n_batch, n_input = input_ids.shape
        assert n_batch == 1, "batch size must be 1"
        n_ctx = n_input + max_new_tokens

        cache = DynamicCache()
        all_tokens = torch.clone(input_ids)  # [1, n_past + n_token]
        while True:
            n_past = cache.get_seq_length()
            print(f"=== {n_past} ===")

            # Input
            in_tokens = all_tokens[:, n_past:]
            n_token = in_tokens.shape[1]  # [1, n_token]
            attention_mask = self.prepare_attention_mask(n_past, n_token)
            position_ids = self.prepare_position_ids(n_past, n_token)

            # Forward
            output = self.model(
                input_ids=in_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                return_dict=True,
                logits_to_keep=1,
            )

            # Output
            out_tokens = torch.argmax(output.logits, dim=-1)  # [1, 1]
            all_tokens = torch.cat((all_tokens, out_tokens), dim=-1)

            # Update
            cache: DynamicCache = output.past_key_values
            cache.crop(all_tokens.shape[1] - 1)

            if self.is_finished(all_tokens, n_ctx, out_tokens):
                return all_tokens

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
        pos = torch.arange(n_past, n_token + n_past, device=self.device)
        pos.unsqueeze_(0)
        return pos
