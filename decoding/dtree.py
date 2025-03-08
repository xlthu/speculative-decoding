from typing import Callable
import torch

__all__ = ["Node", "DraftTree"]


class Node:
    def __init__(self, token: int):
        self.token: int = token
        self.idx: int = -1
        self.pos: int = -1
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
    def __init__(self):
        self.root = Node(-1)  # Special root node
        self._tokens: list[int] = []
        self._pos: list[int] = []

    def new_node(self, token: int) -> Node:
        return Node(token)

    def done(self) -> "DraftTree":
        idx = 0

        def visit(cur: Node, parent: Node):
            if parent is None:
                return  # use default values for root

            # token
            self._tokens.append(cur.token)

            # idx
            nonlocal idx
            cur.idx = idx
            idx += 1

            # pos
            cur.pos = parent.pos + 1
            self._pos.append(cur.pos)

        self.dfs(visit)

        return self

    def size(self) -> int:
        return len(self._tokens)

    def position_ids(self, start: int, device) -> torch.Tensor:
        # [1, size]
        return (
            torch.tensor(self._pos, dtype=torch.long, device=device) + start
        ).unsqueeze(0)

    def tokens(self, device) -> torch.Tensor:
        # [1, size]
        return torch.tensor(self._tokens, dtype=torch.long, device=device).unsqueeze(0)

    def zero_mask(self, mask: torch.Tensor):
        # mask: [size, size]
        assert mask.shape == (self.size(), self.size())

        parents: list[Node] = []

        def visit(cur: Node, parent: Node):
            if parent is not None:
                nonlocal parents
                parents.append(cur)

                for p in parents:
                    mask[cur.idx, p.idx] = 0.0

        def post_visit(cur: Node, parent: Node):
            if parent is not None:
                nonlocal parents
                parents.pop()

        self.dfs(visit, post_visit)

    @staticmethod
    def do_dfs(
        visit: Callable[[Node, Node], None],
        post_visit: Callable[[Node, Node], None],
        cur: Node,
        parent: Node = None,
    ):
        visit(cur, parent)

        for child in cur.children:
            DraftTree.do_dfs(visit, post_visit, child, cur)

        post_visit(cur, parent)

    def dfs(
        self,
        visit: Callable[[Node, Node], None],
        post_visit: Callable[[Node, Node], None] = lambda cur, parent: None,
    ):
        self.do_dfs(visit, post_visit, self.root)

    def debug(self):
        indent = 0

        def visit(cur: Node, parent: Node):
            nonlocal indent
            print(" " * indent, cur)
            indent += 4

        def post_visit(cur: Node, parent: Node):
            nonlocal indent
            indent -= 4

        print("[tree]")
        self.dfs(visit, post_visit)

        print(f"[tokens] {self._tokens}")
        print(f"[pos]    {self._pos}")
