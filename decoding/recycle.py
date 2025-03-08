import torch
from transformers import DynamicCache

from .base import Base
from .dtree import DraftTree
from .utils import tree_attention_mask, tree_position_ids

__all__ = ["Recycle"]


class Recycle(Base):
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
        # return DraftTree().done()

        # For test
        dtree = DraftTree()

        n = [dtree.new_node(i) for i in range(7)]

        dtree.root.add(n[0])
        n[0].add(n[1])

        # n[1].add(n[2])
        # n[1].add(n[3])

        # dtree.root.add(n[4])
        # n[4].add(n[5])
        # n[5].add(n[6])

        return dtree.done()

    ### Output

    def obtain_output(
        self,
        in_tokens: torch.Tensor,
        logits: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        # Verify drafts
        out_tokens, dr_idx = self.verify(in_tokens, logits, self.dtree)
        print(f"{out_tokens.shape}")

        # Update kv cache
        n_reserved = cache.get_seq_length() - self.dtree.size()  # Remove draft tokens
        self.update_kv_cache(cache, n_reserved, dr_idx)
        return out_tokens

    def verify(
        self, in_tokens: torch.Tensor, logits: torch.Tensor, dtree: DraftTree
    ) -> tuple[torch.Tensor, list[int]]:
        # AR, for test
        n_dr = dtree.size()
        out_tokens = torch.argmax(
            logits[:, -n_dr - 1 : -n_dr or None, :], dim=-1
        )  # [1, 1]
        dr_idx = []

        return out_tokens, dr_idx
