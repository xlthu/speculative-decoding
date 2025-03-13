import torch

from .base import Base
from .dtree import *
from .utils import tree_attention_mask, tree_position_ids
from .cache import DynamicCache

__all__ = ["Eagle"]


class Eagle(Base):
    def __init__(self, model):
        super().__init__(model)

    ### Input

    def get_input(
        self, all_tokens: torch.Tensor, cache: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_past = cache.get_seq_length()

        # Decided
        dc_tokens = all_tokens[:, n_past:]
        n_dc = dc_tokens.shape[1]

        # Drafted
        dtree = self.draft(all_tokens)
        dr_tokens = dtree.tokens(self.device)

        # Input
        in_tokens = torch.cat((dc_tokens, dr_tokens), dim=-1)
        attention_mask = tree_attention_mask(
            n_past, n_dc, dtree, self.dtype, self.device
        )
        position_ids = tree_position_ids(n_past, n_dc, dtree, self.device)

        # Record
        self.dtree = dtree

        return in_tokens, attention_mask, position_ids

    def draft(self, all_tokens: torch.Tensor) -> DraftTree:
        # TODO
        return DraftTree().done()

    ### Output

    def obtain_output(
        self,
        in_tokens: torch.Tensor,
        logits: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        # Get output
        out_tokens, dr_idx = self.verify(in_tokens, logits, self.dtree)

        # Update kv cache
        n_reserved = cache.get_seq_length() - self.dtree.size()  # Remove draft tokens
        dr_idx = [n_reserved + i for i in dr_idx]
        self.update_kv_cache(cache, n_reserved, dr_idx)

        return out_tokens

    def verify(
        self, in_tokens: torch.Tensor, logits: torch.Tensor, dtree: DraftTree
    ) -> tuple[torch.Tensor, list[int]]:
        # Verify
        chain = dtree.longest_acc_chain_gd(logits)

        # Output
        out_tokens = [vt.token for vt in chain]
        dr_idx = [vt.idx for vt in chain[1:]]

        out_tokens = torch.tensor(
            out_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        return out_tokens, dr_idx
