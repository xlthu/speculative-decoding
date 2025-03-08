from typing import Callable
import torch
from transformers import DynamicCache

from .base import Base

__all__ = ["Recycle"]


class Node:
    def __init__(self, token: int):
        self.token: int = token
        self.idx: int = 0
        self.pos: int = 0
        self.children: list[Node] = []


class DraftTree:
    def __init__(self):
        self.root = Node(-1)  # Special root node
        self._tokens: list[int] = []

    def new_node(self, token: int):
        return Node(token)

    def done(self):
        pass

    def size(self):
        return len(self._tokens)

    def position_ids(self, start: int, device) -> torch.Tensor:
        pass

    def tokens(self, device) -> torch.Tensor:
        return torch.tensor(self._tokens, dtype=torch.long, device=device)

    @staticmethod
    def do_bfs(visit: Callable[[Node, Node], None], cur: Node, parent: Node = None):
        visit(cur, parent)

        for child in cur.children:
            DraftTree.do_bfs(visit, child, cur)

    def bfs(self, visit: Callable[[Node, Node], None]):
        self.do_bfs(visit, self.root)


class Recycle(Base):
    def __init__(self, model):
        super().__init__(model)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs):
        n_batch, n_input = input_ids.shape
        assert n_batch == 1, "batch size must be 1"
        n_ctx = n_input + max_new_tokens

        cache = DynamicCache()
        all_tokens = torch.clone(input_ids)  # [1, n]
        while True:
            n_past = cache.get_seq_length()
            print(f"=== {n_past} ===")

            # Draft
            dtree = self.draft(all_tokens)

            # Verify - Forward
            dc_tokens = all_tokens[:, n_past:]
            dr_tokens = dtree.tokens(self.device)
            in_tokens = torch.cat((dc_tokens, dr_tokens), dim=-1)

            attention_mask = self.prepare_attention_mask(n_past, dc_tokens, dtree)
            position_ids = self.prepare_position_ids(n_past, dc_tokens, dtree)

            output = self.model(
                input_ids=in_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                return_dict=True,
            )

            # Verify - Output
            out_tokens = self.verify(in_tokens, output.logits, dtree)
            print(f"{out_tokens.shape}")

            # Update - Policy
            self.update(in_tokens, output.logits, dtree)

            # Update - all_tokens & kv cache
            all_tokens = torch.cat((all_tokens, out_tokens), dim=-1)

            cache: DynamicCache = output.past_key_values
            cache.crop(all_tokens.shape[1] - 1)

            # Stop if finished
            if self.is_finished(all_tokens, n_ctx, out_tokens):
                return all_tokens

    def prepare_attention_mask(
        self, n_past: int, dc_tokens: torch.Tensor, dtree: DraftTree
    ):
        # return attention_mask: [1, 1, n_token, n_past + n_token]
        # 0: allow attention, -inf: not allow attention
        n_dc = dc_tokens.shape[1]
        min_dtype = torch.finfo(self.dtype).min
        mask = torch.full(
            (n_dc, n_past + n_dc),
            fill_value=min_dtype,
            dtype=self.dtype,
            device=self.device,
        )
        mask = torch.triu(mask, diagonal=n_past + 1)
        mask = mask.reshape(1, 1, *mask.shape)

        return mask

    def prepare_position_ids(
        self, n_past: int, dc_tokens: torch.Tensor, dtree: DraftTree
    ):
        # return position_ids: [1, n_token]
        n_dc = dc_tokens.shape[1]
        pos = torch.arange(n_past, n_dc + n_past, device=self.device)
        pos.unsqueeze_(0)
        return pos

    def draft(self, all_tokens: torch.Tensor) -> DraftTree:
        return DraftTree()

    def verify(
        self, in_tokens: torch.Tensor, logits: torch.Tensor, dtree: DraftTree
    ) -> torch.Tensor:
        return torch.argmax(logits[:, -1:, :], dim=-1)  # [1, 1]

    def update(self, in_tokens: torch.Tensor, logits: torch.Tensor, dtree: DraftTree):
        pass
