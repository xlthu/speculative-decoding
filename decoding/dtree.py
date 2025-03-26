from typing import Callable
import torch
from dataclasses import dataclass
import random
import numpy as np

__all__ = ["Node", "VerifiedToken", "DraftTree"]


class Node:
    def __init__(self, token: int, score: float = 0.0, parent: "Node" = None):
        self.token: int = token
        self.score: float = score

        self.idx: int = -1  # Index in the flatten represetation
        self.pos: int = -1  # Real position in the text / Tree depth

        self.parent = parent
        self.children: list[Node] = []

    def add_child(self, child: "Node"):
        assert child.parent is None or child.parent == self
        child.parent = self
        self.children.append(child)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"token={self.token}, "
            f"score={self.score}, "
            f"idx={self.idx}, "
            f"pos={self.pos}, "
            f"parent={self.parent.idx if self.parent else self.parent}, "
            f"children={[c.idx for c in self.children]}"
            f")"
        )


@dataclass
class VerifiedToken:
    token: int
    idx: int


class DraftTree:
    """Draft Tree

    1. Root is a special node, does not represent any token.
    2. Flatten representation is obtained by DFS.
    """

    def __init__(self, root_token: int = -1):
        self.root = Node(root_token)
        self._tokens: list[int] = []  # Flatten tokens
        self._pos: list[int] = []  # Flatten tositions

    def done(self) -> "DraftTree":
        """Called when all nodes are added"""
        idx = 0

        def visit(cur: Node, parent: Node):
            if parent is not None:  # use default values for root
                # token
                self._tokens.append(cur.token)

                # idx
                nonlocal idx
                cur.idx = idx
                idx += 1

                # pos
                cur.pos = parent.pos + 1
                self._pos.append(cur.pos)
            return True

        self.dfs(visit)

        return self

    def size(self) -> int:
        """#draft tokens"""
        return len(self._tokens)

    def position_ids(self, start: int, device) -> torch.Tensor:
        """Flatten positions"""
        # [1, size]
        return (
            torch.tensor(self._pos, dtype=torch.long, device=device) + start
        ).unsqueeze(0)

    def tokens(self, device) -> torch.Tensor:
        """Flatten tokens"""
        # [1, size]
        return torch.tensor(self._tokens, dtype=torch.long, device=device).unsqueeze(0)

    def zero_mask(self, mask: torch.Tensor):
        """Zeroing the attention position in the mask, in place

        Args:
            mask (torch.Tensor): `-inf` tensor of [size(), size()]
        """
        # mask: [size, size]
        assert mask.shape == (self.size(), self.size())

        parents: list[Node] = []

        def visit(cur: Node, parent: Node):
            if parent is not None:
                nonlocal parents
                parents.append(cur)

                for p in parents:
                    mask[cur.idx, p.idx] = 0.0
            return True

        def post_visit(cur: Node, parent: Node):
            if parent is not None:
                nonlocal parents
                parents.pop()

        self.dfs(visit, post_visit)

    def longest_acc_chain_gd(self, logits: torch.Tensor) -> list[VerifiedToken]:
        """Longest accepted chain using greedy decoding

        Args:
            logits (torch.Tensor): [1, ..., n_vocab]
                logits from original model

        Returns:
            list[VerifiedToken]: [1, 1 + size()]
        """
        assert logits.shape[1] >= self.size() + 1

        logits = logits[0, -self.size() - 1 :]
        out_tokens = torch.argmax(logits, dim=-1).tolist()  # [1 + size]

        chain = []
        cur = self.root
        done = False
        while not done:
            out_token = out_tokens[cur.idx + 1]
            chain.append(VerifiedToken(out_token, cur.idx))

            done = True
            for child in cur.children:
                if child.token == out_token:
                    cur = child
                    done = False
                    break

        return chain

    def longest_acc_chain_topk(self, logits: torch.Tensor, k=2) -> list[VerifiedToken]:
        """Longest accepted chain using top-k acceptance

        Args:
            logits (torch.Tensor): [1, ..., n_vocab]
                logits from original model

        Returns:
            list[VerifiedToken]: [1, 1 + size()]
        """
        assert logits.shape[1] >= self.size() + 1

        logits = logits[0, -self.size() - 1 :]
        _, out_tokens = torch.topk(logits, k)
        out_tokens = out_tokens.tolist()  # [1 + size, k]

        def get_matched_child(token: int, cur: Node):
            for child in cur.children:
                if child.token == token:
                    return child
            return None

        chain = []
        cur = self.root
        done = False
        while not done:
            cur_out_tokens = out_tokens[cur.idx + 1]
            vt = VerifiedToken(cur_out_tokens[0], cur.idx)

            done = True
            for token in cur_out_tokens:
                child = get_matched_child(token, cur)
                if child is not None:
                    vt.token = token
                    cur = child
                    done = False
                    break

            chain.append(vt)

        return chain

    def longest_acc_chain_mrss(
        self, logits: torch.Tensor, last_step_sample=True
    ) -> list[VerifiedToken]:
        """Longest accepted chain using (modified) multi-round speculative sampling

            See EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty,
                Appendix A.2 Algorithm 1

            Modification: sample from original logits in the last step,
                rather than from (original_logits - draft_logits)

        Args:
            logits (torch.Tensor): [1, 1 + size(), n_vocab]
                logits from original model
            last_step_sample (bool): sample or greedy in the last step

        Returns:
            list[VerifiedToken]: [1, 1 + size()]
        """
        assert logits.shape[1] >= self.size() + 1

        probs = torch.softmax(logits[0, -self.size() - 1 :].float(), dim=-1)

        chain = []
        cur = self.root
        done = False
        while not done:
            done = True
            for child in cur.children:
                p = probs[cur.idx + 1, child.token].item()
                dp = np.exp(child.score - cur.score)
                if random.random() < p / dp:
                    chain.append(VerifiedToken(child.token, cur.idx))
                    cur = child
                    done = False
                    break

        if last_step_sample:
            p = probs[cur.idx + 1].cpu().numpy()
            token = np.random.choice(logits.shape[-1], p=p)
        else:
            token = torch.argmax(probs[cur.idx + 1], dim=-1)
        chain.append(VerifiedToken(token, cur.idx))

        return chain

    @staticmethod
    def do_dfs(
        visit: Callable[[Node, Node], bool],
        post_visit: Callable[[Node, Node], None],
        cur: Node,
        parent: Node = None,
    ):
        if not visit(cur, parent):
            return

        for child in cur.children:
            DraftTree.do_dfs(visit, post_visit, child, cur)

        post_visit(cur, parent)

    def dfs(
        self,
        visit: Callable[[Node, Node], bool],
        post_visit: Callable[[Node, Node], None] = lambda cur, parent: None,
    ):
        """DFS

        Args:
            visit (Callable[[Node, Node], bool]):
                Called when first visit a node
                    Args: the node to be visited, parent of the node
                    Returns: to visit the sub tree of the node or not
            post_visit (Callable[[Node, Node], None], optional):
                Called after all the nodes in the sub-tree are visited
                    Args: the node whose sub-tree is visited, parent of the node
                    Retuns: ignored

        Notes:
            If `visit` return False, then `post_visit` will not called on this node
        """
        self.do_dfs(visit, post_visit, self.root)

    def debug(self, print_func=print):
        """Print a tree representation"""
        indent = 0

        def visit(cur: Node, parent: Node):
            nonlocal indent
            print_func(" " * indent, cur)
            indent += 4
            return True

        def post_visit(cur: Node, parent: Node):
            nonlocal indent
            indent -= 4

        print_func("[tree]")
        self.dfs(visit, post_visit)

        print_func(f"[tokens] {self._tokens}")
        print_func(f"[pos]    {self._pos}")
