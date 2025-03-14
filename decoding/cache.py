import transformers
import torch

__all__ = ["DynamicCache"]


class DynamicCache(transformers.DynamicCache):
    def __init__(self):
        super().__init__()
        self.hidden: torch.Tensor = None

    def update_hidden(self, hidden: torch.Tensor):
        if self.hidden is None:
            self.hidden = hidden
        else:
            self.hidden = torch.cat((self.hidden, hidden), dim=-2)

    def crop(self, max_length: int):
        assert False
        super().crop(max_length)

        if self.hidden is not None:
            self.hidden = self.hidden[..., :max_length, :]

    def pick(self, n_reserved: int, picked: list[int] = None):
        """Pick KV Cache lines

            First, pick `n_reserved` lines from the beginning of `cache`.
            Then, pick the lines indexed by `picked` and append them to `cache`.

        Args:
            n_reserved (int): #Reserved lines from the beginning
            picked (list[int]): Indecics of lines to pick after the reserved lines
        """

        if picked:
            pt = torch.tensor(picked, dtype=torch.long, device=self.hidden.device)

            def update(tensor: torch.Tensor):
                return torch.cat(
                    (tensor[..., :n_reserved, :], tensor.index_select(-2, pt)),
                    dim=-2,
                )

        else:

            def update(tensor: torch.Tensor):
                return tensor[..., :n_reserved, :]

        for lid in range(len(self.key_cache)):
            # [batch_size, num_heads, seq_len, head_dim]
            self.key_cache[lid] = update(self.key_cache[lid])
            self.value_cache[lid] = update(self.value_cache[lid])

        if self.hidden is not None:
            self.hidden = update(self.hidden)

        self._seen_tokens = n_reserved + len(picked or [])
