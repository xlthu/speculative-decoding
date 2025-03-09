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

        tokens = [95456, 0, 362, 3460, 4128, 1614, 374, 264, 943]
        tokens = [95456, 0, 362, 3460, 4128, 1614]
        tokens = [95456, 1, 2, 3, 4, 5, 6, 7]

        p = dtree.root
        for t in tokens:
            n = dtree.new_node(t)
            p.add(n)
            p = n

        return dtree.done()

    ### Output

    def obtain_output(
        self,
        in_tokens: torch.Tensor,
        logits: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        # Get output
        out_tokens, dr_idx = self.verify(in_tokens, logits, self.dtree)
        print(f"{out_tokens.shape}")

        # Update kv cache
        n_reserved = cache.get_seq_length() - self.dtree.size()  # Remove draft tokens
        dr_idx = [n_reserved + i for i in dr_idx]
        self.update_kv_cache(cache, n_reserved, dr_idx)
        return out_tokens

    def verify(
        self, in_tokens: torch.Tensor, logits: torch.Tensor, dtree: DraftTree
    ) -> tuple[torch.Tensor, list[int]]:
        n_dr = dtree.size()

        # Greedy decoding
        out_tokens = torch.argmax(logits[0, -n_dr - 1 :], dim=-1)  # [1 + n_dr]

        if n_dr == 0:
            return out_tokens.unsqueeze(0), []

        # Verify draft
        eq = in_tokens[0, -n_dr:] == out_tokens[:-1]  # [n_dr]
        longest_acc_chain = dtree.longest_acc_chain(eq.tolist())

        # Output
        dr_idx = [n.idx for n in longest_acc_chain]
        out_tokens = [out_tokens[0].item()] + [
            out_tokens[n.idx + 1].item() for n in longest_acc_chain
        ]

        out_tokens = torch.tensor(
            out_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        return out_tokens, dr_idx
