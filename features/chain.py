#!/usr/bin/env python3

import re
from typing import Callable, Iterable, TypeVar, TypeVarTuple

from joblib import Parallel, delayed

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
R = TypeVar("R")


class Chain:
    """
    Provides facilities for querying LLMs and parsing response.
    """

    def __init__(self, verbose: bool = False):
        self.pool = Parallel(n_jobs=-1, prefer="threads", verbose=10 if verbose else 0)

    def _parse(self, results: list[str], prefix: str = "") -> list[str]:
        return [
            g.strip()
            for r in results
            for p in [re.search(prefix + r"(?:\s*)(?P<res>.*)", r)]
            for g in [p and p.group("res")]
            if p and g
        ]

    def _pmap(
        self, meth: Callable[[T, *Ts], R], it: Iterable[T], *args: *Ts
    ) -> list[R]:
        return self._parallel(meth, [(x, *args) for x in it])

    def _parallel(self, meth: Callable[[*Ts], R], it: Iterable[tuple[*Ts]]) -> list[R]:
        return self.pool(delayed(meth)(*x) for x in it) or []
