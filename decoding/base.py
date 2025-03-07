import torch
from torch import nn

__all__ = ["Base"]


class Base(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.config.torch_dtype

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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # Utility

    def get_eos_token_ids(self):
        eos_token_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        return eos_token_ids

    def is_eos(self, token: int):
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
