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
        super().crop(max_length)

        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        if self.hidden is not None:
            self.hidden = self.hidden[..., :max_length, :]
