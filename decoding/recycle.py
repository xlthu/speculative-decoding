import torch
from transformers import DynamicCache

from .base import Base
from .dtree import *
from .utils import tree_attention_mask, tree_position_ids

__all__ = ["Recycle"]


class Recycle(Base):
    def __init__(self, model, n_vocab: int, k: int = 8):
        super().__init__(model)
        self.adj_mat = [[-1] * k for _ in range(n_vocab)]
        self.k = k

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
        # Chain
        # layers = [
        #     [1],
        #     [1],
        #     [1],
        #     [1],
        #     [1],
        # ]

        # Eagle-1 tree
        layers = [
            [4],
            [3, 2, 2, 1],
            [3, 2, 2, 1, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0],
        ]

        # Token Recycling Tree
        # layers = [
        #     [8],
        #     [8, 4, 3, 2, 1, 1, 1, 1],
        #     [8, 3, 2, 1, 1, 1, 1, 1, 2, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
        #     [5, 2, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        #     [3, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        #     [2, 0, 0, 1, 0, 0, 0, 0],
        # ]

        dtree = DraftTree()
        dtree.root.token = all_tokens[0, -1].item()

        # Fill according to static tree
        cur_layer = [dtree.root]
        for depth, layer in enumerate(layers):
            # fmt: off
            assert len(layer) == len(cur_layer), f"{depth}: {len(layer)} != {len(cur_layer)}"
            # fmt: on

            new_layer = []
            for nchild, parent in zip(layer, cur_layer):
                parent.children = [Node(t) for t in self.adj_mat[parent.token][:nchild]]
                new_layer.extend(parent.children)
            cur_layer = new_layer

        # Remove -1 token
        def remove_invalid_token(cur: Node, parent: Node):
            cur.children = [c for c in cur.children if c.token != -1]
            return True

        dtree.dfs(remove_invalid_token)

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

        # Update kv cache
        n_reserved = cache.get_seq_length() - self.dtree.size()  # Remove draft tokens
        dr_idx = [n_reserved + i for i in dr_idx]
        self.update_kv_cache(cache, n_reserved, dr_idx)

        # Update adjacency matrix
        self.update_adj_mat(in_tokens, logits)
        return out_tokens

    def verify(
        self, in_tokens: torch.Tensor, logits: torch.Tensor, dtree: DraftTree
    ) -> tuple[torch.Tensor, list[int]]:
        n_dr = dtree.size()

        # Greedy decoding
        out_tokens = torch.argmax(logits[0, -n_dr - 1 :], dim=-1).tolist()  # [1 + n_dr]

        # Verify draft
        longest_acc_chain = dtree.longest_acc_chain(out_tokens)

        # Output
        dr_idx = [n.idx for n in longest_acc_chain]
        out_tokens = [out_tokens[0]] + [
            out_tokens[n.idx + 1] for n in longest_acc_chain
        ]

        out_tokens = torch.tensor(
            out_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        return out_tokens, dr_idx

    def update_adj_mat(self, in_tokens: torch.Tensor, logits: torch.Tensor):
        _, indices = torch.topk(logits[0], self.k)
        indices = indices.tolist()

        for token, idx in zip(in_tokens[0].tolist(), indices):
            self.adj_mat[token] = idx
