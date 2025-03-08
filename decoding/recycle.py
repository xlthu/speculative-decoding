import torch
from transformers import DynamicCache

from .base import Base
from .dtree import *

__all__ = ["Recycle"]


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
            out_tokens, dr_idx = self.verify(in_tokens, output.logits, dtree)
            print(f"{out_tokens.shape}")

            # Update
            cache: DynamicCache = output.past_key_values
            self.update_kv_cache(cache, n_past, dc_tokens, dr_idx)

            all_tokens = torch.cat((all_tokens, out_tokens), dim=-1)

            # Stop if finished
            if self.is_finished(all_tokens, n_ctx, out_tokens):
                return all_tokens

    def prepare_attention_mask(
        self, n_past: int, dc_tokens: torch.Tensor, dtree: DraftTree
    ):
        # return attention_mask: [1, 1, n_dc + n_dr, n_past + n_dc + n_dr]
        # 0: allow attention, -inf: not allow attention
        n_dc = dc_tokens.shape[1]
        n_dr = dtree.size()
        min_dtype = torch.finfo(self.dtype).min

        lmask = torch.full(
            size=(n_dc + n_dr, n_past + n_dc),
            fill_value=min_dtype,
            dtype=self.dtype,
            device=self.device,
        )
        lmask = torch.triu(lmask, diagonal=n_past + 1)

        rmask = torch.full(
            size=(n_dc + n_dr, n_dr),
            fill_value=min_dtype,
            dtype=self.dtype,
            device=self.device,
        )
        dr_mask = rmask[n_dc:, :]
        dtree.zero_mask(dr_mask)

        mask = torch.cat((lmask, rmask), dim=-1)
        mask = mask.reshape(1, 1, *mask.shape)

        return mask

    def prepare_position_ids(
        self, n_past: int, dc_tokens: torch.Tensor, dtree: DraftTree
    ):
        # return position_ids: [1, n_past + n_dc + n_dr]
        n_dc = dc_tokens.shape[1]
        dc_pos = torch.arange(n_past, n_past + n_dc, device=self.device).unsqueeze(0)
        dr_pos = dtree.position_ids(n_past + n_dc, self.device)
        pos = torch.cat((dc_pos, dr_pos), dim=-1)
        return pos

    def draft(self, all_tokens: torch.Tensor) -> DraftTree:
        # return DraftTree().done()

        # For test
        dtree = DraftTree()

        n = [dtree.new_node(i) for i in range(7)]

        dtree.root.add(n[0])
        n[0].add(n[1])

        n[1].add(n[2])
        n[1].add(n[3])

        dtree.root.add(n[4])
        n[4].add(n[5])
        n[5].add(n[6])

        return dtree.done()

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

    def update_kv_cache(
        self,
        cache: DynamicCache,
        n_past: int,
        dc_tokens: torch.Tensor,
        dr_idx: list[int],
    ):
        n_past += dc_tokens.shape[1]
        dr_idx = torch.tensor(dr_idx, dtype=torch.long, device=dc_tokens.device)

        def update(tensor: torch.Tensor):
            return torch.cat(
                (tensor[..., :n_past, :], tensor.index_select(-2, dr_idx)), dim=-2
            )

        for lid in range(len(cache.key_cache)):
            # [batch_size, num_heads, seq_len, head_dim]
            cache.key_cache[lid] = update(cache.key_cache[lid])
            cache.value_cache[lid] = update(cache.value_cache[lid])

        n_past += len(dr_idx)
        cache._seen_tokens = n_past
