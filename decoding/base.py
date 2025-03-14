import torch
from torch import nn

from .cache import DynamicCache
from .stat import Stat

__all__ = ["Base"]


class Base(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def model_type(self):
        return self.model.config.model_type

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.config.torch_dtype

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        cache: DynamicCache = None,
        stat: Stat = None,
    ):
        n_batch, n_input = input_ids.shape
        assert n_batch == 1, "batch size must be 1"
        n_ctx = n_input + max_new_tokens

        cache = cache or DynamicCache()
        stat = stat or Stat()
        all_tokens = torch.clone(input_ids)  # [1, n]

        while True:
            # Input
            with stat.tik_tok("get_input"):
                in_tokens, attention_mask, position_ids = self.get_input(
                    all_tokens, cache
                )

            # Forward
            with stat.tik_tok("forward"):
                output = self.model(
                    input_ids=in_tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # Output
            cache = output.past_key_values
            cache.update_hidden(output.hidden_states[-1])
            assert cache.get_seq_length() == cache.hidden.shape[-2]

            with stat.tik_tok("obtain_output"):
                out_tokens = self.obtain_output(in_tokens, output.logits, cache)
            stat.put("acc_len", out_tokens.shape[1])

            all_tokens = torch.cat((all_tokens, out_tokens), dim=-1)

            # Stop if finished
            if self.is_finished(all_tokens, n_ctx, out_tokens):
                return {"output_ids": all_tokens, "stat": stat, "cache": cache}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # Hook

    def get_input(
        self, all_tokens: torch.Tensor, cache: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get input and

        Args:
            all_tokens (torch.Tensor): [1, n_past + n_input]
                All tokens including past evaluted and currently inputted tokens
            cache (DynamicCache): KV cache of all evaluted tokens

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                in_tokens: [1, n_in = n_dc + n_dr]
                attention_mask: [1, 1, n_in, n_past + n_in]
                position_ids: [1, n_in]
        """
        raise NotImplementedError()

    def obtain_output(
        self,
        in_tokens: torch.Tensor,
        logits: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        """Get output from logits and update kv cache

        Args:
            n_past (int): #Tokens
            in_tokens (torch.Tensor): [1, n_in], Tokens evaluated
            logits (torch.Tensor): [1, n_in, n_vocab], Logits from evaluating `in_tokens`
            cache (DynamicCache): KV cache after evaluating `in_tokens`

        Returns:
            out_tokens: [1, n_out]
        """
        raise NotImplementedError()

    # Utility

    def get_eos_token_ids(self) -> list[int]:
        """Get all eos tokens"""
        eos_token_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        return eos_token_ids

    def is_eos(self, token: int) -> bool:
        """Check if token is eos token"""
        return token in self.get_eos_token_ids()

    def is_finished(
        self, all_tokens: torch.Tensor, n_ctx: int, out_tokens: torch.Tensor
    ):
        """Is generation finished?

        Args:
            all_tokens (torch.Tensor): All tokens including inputs and outputs [1, n_seq]
            n_ctx (int): #Max n_seq
            out_tokens (torch.Tensor): [1, n_out]
        """

        if all_tokens.shape[1] >= n_ctx:
            return True

        for token in out_tokens[0]:
            if self.is_eos(token):
                return True

        return False
