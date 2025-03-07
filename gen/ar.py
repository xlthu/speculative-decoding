import torch
from transformers import DynamicCache

from .base import Generator

__all__ = ["AutoRegressive"]


class AutoRegressive(Generator):
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs):
        batch_size, input_len = input_ids.shape
        assert batch_size == 1, "batch_size must be 1"
        n_ctx = input_len + max_new_tokens

        cache = DynamicCache()
        tokens = torch.clone(input_ids)
        while True:
            n_past = cache.get_seq_length()
            print(f"=== {n_past} ===")
            cur_tokens = tokens[:, n_past:]
            n_token = cur_tokens.shape[1]

            # Forward
            output = self.model(
                input_ids=cur_tokens,
                attention_mask=self.prepare_attention_mask(n_past, n_token),
                position_ids=self.prepare_position_ids(n_past, n_token),
                past_key_values=cache,
                return_dict=True,
                logits_to_keep=1,
            )

            # Output
            logits = output.logits
            gen_ids = torch.argmax(logits, dim=-1)
            tokens = torch.cat((tokens, gen_ids), dim=-1)

            if self.is_finished(tokens, n_ctx, gen_ids):
                return tokens

            # Update
            cache: DynamicCache = output.past_key_values

    def is_eos(self, token: int):
        return token in self.get_eos_token_ids()

    def is_finished(self, tokens: torch.Tensor, n_ctx: int, gen_ids: torch.Tensor):
        if tokens.shape[1] >= n_ctx:
            return True

        for token in gen_ids[0]:
            if self.is_eos(token):
                return True

        return False

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
