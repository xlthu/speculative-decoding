import time
from contextlib import contextmanager

__all__ = ["Stat"]


class Stat:
    def __init__(self):
        self.stat: dict[str, list] = {}
        self.start: dict[str, int] = {}

    def __repr__(self) -> str:
        return str(self.stat)

    def tik(self, label: str):
        self.start[label] = time.time_ns()

    def tok(self, label: str) -> float:
        elapsed = (time.time_ns() - self.start[label]) / 1_000_000  # to milliseconds
        del self.start[label]
        self.put(label, elapsed)
        return elapsed

    @contextmanager
    def tik_tok(self, label: str):
        self.tik(label)
        yield self
        self.tok(label)

    def put(self, label: str, value):
        if label not in self.stat:
            self.stat[label] = []
        self.stat[label].append(value)

    def __getitem__(self, label: str):
        return self.stat[label]

    def __iter__(self):
        return iter(self.stat)

    def labels(self):
        return self.stat.keys()

    def to_dict(self):
        return self.stat

    @classmethod
    def from_dict(cls, d: dict):
        s = cls()
        s.stat = d
        return s
