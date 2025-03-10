from typing import Callable
import torch

__all__ = ["Node", "DraftTree"]


class Node:
    def __init__(self, token: int):
        self.token: int = token
        self.idx: int = -1  # Index in the flatten represetation
        self.pos: int = -1  # Real position in the text / Tree depth
        self.children: list[Node] = []

    def add(self, child: "Node"):
        self.children.append(child)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"token={self.token}, "
            f"idx={self.idx}, "
            f"pos={self.pos}, "
            f"children={[c.idx for c in self.children]}"
            f")"
        )


class DraftTree:
    """Draft Tree

    1. Root is a special node, does not represent any token.
    2. Flatten representation is obtained by DFS.
    """

    def __init__(self):
        self.root = Node(-1)  # Special root node
        self._tokens: list[int] = []  # Flatten tokens
        self._pos: list[int] = []  # Flatten tositions

    def new_node(self, token: int) -> Node:
        """Use this to create node (for C++, recycle the memory)"""
        return Node(token)

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
            mask (torch.Tensor): `-inf` tensor of [size, size]
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

    def longest_acc_chain(self, out_tokens: list[int]) -> list[Node]:
        """Get the longest accepted node chain

        Args:
            out_tokens (list[int]): Output tokens correspnding to [root, self.tokens...]

        Returns:
            list[Node]: The longest accepted node chain
        """
        assert len(out_tokens) == len(self._tokens) + 1

        longest_chain = []
        cur_chain = []

        def visit(cur: Node, parent: Node):
            if parent is not None:
                nonlocal cur_chain, longest_chain

                if cur.token != out_tokens[parent.idx + 1]:
                    # Not visit the sub tree if not accepted
                    return False

                cur_chain.append(cur)
                longest_chain = (
                    longest_chain
                    if len(longest_chain) >= len(cur_chain)
                    else cur_chain.copy()
                )
            return True

        def post_visit(cur: Node, parent: Node):
            if parent is not None:
                nonlocal cur_chain
                cur_chain.pop()

        self.dfs(visit, post_visit)

        return longest_chain

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

    def debug(self):
        """Print a tree representation"""
        indent = 0

        def visit(cur: Node, parent: Node):
            nonlocal indent
            print(" " * indent, cur)
            indent += 4
            return True

        def post_visit(cur: Node, parent: Node):
            nonlocal indent
            indent -= 4

        print("[tree]")
        self.dfs(visit, post_visit)

        print(f"[tokens] {self._tokens}")
        print(f"[pos]    {self._pos}")
