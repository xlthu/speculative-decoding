import torch

from .base import Base
from .eagle_model import EAModel
from .dtree import *
from .utils import tree_attention_mask, tree_position_ids
from .cache import DynamicCache

__all__ = ["Eagle"]

import contextlib


class Log:
    def __init__(self):
        self.prefix = ""

    def push(self, step=2):
        self.prefix += "|" + " " * step

    def pop(self, step=2):
        self.prefix = self.prefix[: -(step + 1)]

    def log(self, *args):
        if False:
            print(f"{self.prefix}", end="")
            print(*args)

    @contextlib.contextmanager
    def scope(self, label: str):
        self.enter_scope(label)
        yield
        self.exit_scope(label)

    def enter_scope(self, label: str):
        self.log(f"[{label}]")
        self.push()

    def exit_scope(self, label: str):
        self.pop()
        self.log(f"[{label}]")


logger = Log()


class Eagle(Base):
    def __init__(self, model, draft_model: EAModel, h: int, k: int, m: int):
        """Eagle

        Args:
            model (Huggingface Models): Main model
            draft_model (EAModel): Eagle draft model
            h (int): Draft tree depth
            k (int): Top-k ranking
            m (int): Total draft tokens
        """
        super().__init__(model)
        self.dm = draft_model
        self.dm_cache = DynamicCache()
        self.h = h
        self.k = k
        self.m = m

    ### Input

    def get_input(
        self, all_tokens: torch.Tensor, cache: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_past = cache.get_seq_length()
        logger.log(f"=== {n_past} ===")
        logger.log(f"{all_tokens=}")

        # Decided
        dc_tokens = all_tokens[:, n_past:]
        n_dc = dc_tokens.shape[1]

        # Drafted
        with logger.scope("draft"):
            dtree = self.draft(all_tokens, cache)
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

    def draft(self, all_tokens: torch.Tensor, cache: DynamicCache) -> DraftTree:
        n_past = cache.get_seq_length()
        n_dc = all_tokens.shape[1] - n_past

        logger.log(f"{n_past=}")
        logger.log(f"{n_dc=}")

        if n_dc > 1:
            # Cannot draft due to the missing hidden states of main model
            return DraftTree().done()

        device = all_tokens.device

        def long_tensor(data):
            return torch.tensor(data, dtype=torch.long, device=device)

        # Fill dm kv cache & inference on root (last token of all_tokens)
        start = self.dm_cache.get_seq_length()
        out_hidden, logits = self.dm_forward(
            hidden_states=cache.hidden[:, start:, :],
            in_tokens=all_tokens[:, start + 1 :],
            attention_mask=None,
            position_ids=None,
        )
        dm_n_past_saved = self.dm_cache.get_seq_length()
        logger.log(f"{dm_n_past_saved=}")

        # Init by last token of all_tokens
        dtree = DraftTree(all_tokens[0, -1].item())
        all_hidden = out_hidden[:, -1:, :]
        layer = self.expand(logits[:, -1:, :], [dtree.root])

        # Expansion
        for h in range(self.h):
            with logger.scope(f"===== {h} ====="):
                # Forward
                pidx = long_tensor([n.parent.idx + 1 for n in layer])
                in_hidden = all_hidden.index_select(-2, pidx)
                in_tokens = long_tensor([[n.token for n in layer]])
                attention_mask = self.layer_attention_mask(
                    dm_n_past_saved, layer, self.dm.dtype, device
                )
                position_ids = long_tensor([[h + dm_n_past_saved] * len(layer)])
                logger.log(f"{pidx=}")
                logger.log(f"{in_tokens=}")
                logger.log(f"{position_ids=}")

                out_hidden, logits = self.dm_forward(
                    in_hidden, in_tokens, attention_mask, position_ids
                )

                # Expand
                all_hidden = torch.cat((all_hidden, out_hidden), dim=-2)
                layer = self.expand(logits, layer)

                logger.log(f"{all_hidden.shape=}")
                dtree.debug(logger.log)

        # Reranking
        with logger.scope("rerank"):
            logger.log(f"before:")
            dtree.debug(logger.log)

            self.rerank(dtree)

            logger.log(f"after:")
            dtree.debug(logger.log)

        # Remove all draft tokens
        self.dm_cache.pick(dm_n_past_saved)

        return dtree.done()

    def expand(self, logits: torch.Tensor, layer: list[Node]):
        assert logits.shape[-2] == len(layer)
        with logger.scope("expand"):
            # Populate the next layer with top-k children of each node
            next_layer: list[Node] = []
            for i, parent in enumerate(layer):
                _, out_tokens = torch.topk(logits[0, i], self.k)
                out_tokens = out_tokens.tolist()

                for token in out_tokens:
                    score = logits[0, i, token].item() + parent.score
                    child = Node(token, score, parent)
                    parent.add_child(child)
                    next_layer.append(child)

            logger.log(f"before topk, {next_layer=}")

            # Top-k across all children
            next_layer.sort(key=lambda n: n.score, reverse=True)
            next_layer = next_layer[: self.k]

            logger.log(f"after topk, {next_layer=}")

            # Set idx of top-k nodes for the next dm_forward
            n_past_dr = layer[-1].idx + 1
            for i, child in enumerate(next_layer):
                child.idx = n_past_dr + i

        return next_layer

    def rerank(self, dtree: DraftTree):
        # Collect all nodes
        all_nodes = []

        def collect(cur: Node, parent: Node):
            all_nodes.append(cur)
            return True

        dtree.dfs(collect)

        # Sort
        all_nodes.sort(key=lambda node: (node.score, -node.idx), reverse=True)
        reserved = all_nodes[: self.m + 1]  # include root

        # Remove
        for node in reserved:
            node.children = [child for child in node.children if child in reserved]

        return dtree

    def layer_attention_mask(self, n_past_dc: int, layer: list[Node], dtype, device):
        with logger.scope("layer_attention_mask"):
            n_past_dr = layer[0].idx

            logger.log(f"{n_past_dc=}")
            logger.log(f"{n_past_dr=}")
            logger.log(f"{layer=}")

            n_dr = len(layer)
            min_dtype = torch.finfo(dtype).min

            lmask = torch.full(
                size=(n_dr, n_past_dc),
                fill_value=0.0,
                dtype=dtype,
                device=device,
            )

            rmask = torch.full(
                size=(n_dr, n_past_dr + n_dr),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            for i, node in enumerate(layer):
                p = node
                while p.idx != -1:  # until root (excluded)
                    rmask[i, p.idx] = 0.0
                    p = p.parent

            mask = torch.cat((lmask, rmask), dim=-1)
            mask = mask.reshape(1, 1, *mask.shape)

            return mask

    def dm_forward(
        self,
        hidden_states: torch.Tensor,
        in_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        if attention_mask is not None:
            assert attention_mask.shape == (
                1,
                1,
                in_tokens.shape[-1],
                self.dm_cache.get_seq_length() + in_tokens.shape[-1],
            )
        if position_ids is not None:
            assert position_ids.shape == in_tokens.shape

        with logger.scope("dm_forward"):
            logger.log(f"{in_tokens=}")
            logger.log(f"{hidden_states.shape=}")

            inputs_embeds = self.model.get_input_embeddings()(in_tokens)

            output = self.dm(
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.dm_cache,
                return_dict=True,
            )
            hidden = output.last_hidden_state
            self.dm_cache = output.past_key_values

            # NOTE: adapt to official EAGLE code
            hidden = self.model.model.norm(hidden)

            logits = self.model.get_output_embeddings()(hidden)
            logits = torch.nn.functional.log_softmax(logits, dim=-1)

            return hidden, logits

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
        cache.pick(n_reserved, dr_idx)

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
