import time
from typing import Any, Optional
from dataclasses import dataclass

import numpy as np

@dataclass
class Info:
  converged: bool
  iterations: int
  residual: float
  time: float

class IterativeSolver:
  def __init__(self, tol: float = 1.0e-8, max_iter: int = 20000) -> None:
    self.tol: float = tol
    self.max_iter: int = max_iter

  def solve(self, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x0: Optional[np.typing.NDArray[Any]] = None, tol: Optional[float] = None, max_iter: Optional[int] = None) -> tuple[np.typing.NDArray[Any], Info]:
    t0 = time.perf_counter()
    x, iterations, converged = self._solve(A, b, np.zeros_like(b) if x0 is None else x0.copy(), self.tol if tol is None else tol, self.max_iter if max_iter is None else max_iter)
    info = Info(converged, iterations, self._residual(A, b, x), time.perf_counter() - t0)
    return x, info

  def _residual(self, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x: np.typing.NDArray[Any]) -> float:
    return (np.linalg.norm(A @ x - b) / np.linalg.norm(b)).item()

  def _solve(self, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x0: np.typing.NDArray[Any], tol: float, max_iter: int) -> tuple[np.typing.NDArray[Any], int, bool]:
    raise NotImplementedError

class JacobiSolver(IterativeSolver):
  def _solve(self, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x0: np.typing.NDArray[Any], tol: float, max_iter: int) -> tuple[np.typing.NDArray[Any], int, bool]:
    x = x0
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(0, max_iter):
      x = (b - R @ x) / D
      if self._residual(A, b, x) < tol:
        return x, k + 1, True
    return x, max_iter, False

class GaussSeidelSolver(IterativeSolver):
  def _solve(self, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x0: np.typing.NDArray[Any], tol: float, max_iter: int) -> tuple[np.typing.NDArray[Any], int, bool]:
    x, n = x0, b.size
    D, L, U = np.diag(A), np.tril(A, -1), np.triu(A,  1)
    for k in range(0, max_iter):
      for i in range(n):
        sigma = L[i] @ x + U[i] @ x
        x[i] = (b[i] - sigma) / D[i]
      if self._residual(A, b, x) < tol:
        return x, k + 1, True
    return x, max_iter, False

class GradientDescentSolver(IterativeSolver):
  def _solve(self, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x0: np.typing.NDArray[Any], tol: float, max_iter: int) -> tuple[np.typing.NDArray[Any], int, bool]:
    x, r = x0, b - A @ x0
    if np.linalg.norm(r) > tol * np.linalg.norm(b):
      for k in range(0, max_iter):
        Ar = A @ r
        alpha = (r @ r) / (r @ Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        if self._residual(A, b, x) < tol:
          return x, k + 1, True
      return x, max_iter, False
    return x, 0, True

class ConjugateGradientSolver(IterativeSolver):
  def _solve(self, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x0: np.typing.NDArray[Any], tol: float, max_iter: int) -> tuple[np.typing.NDArray[Any], int, bool]:
    x, r = x0, b - A @ x0
    p, rs_old = r.copy(), r @ r
    for k in range(0, max_iter):
      Ap = A @ p
      alpha = rs_old / (p @ Ap)
      x = x + alpha * p
      r = r - alpha * Ap
      if self._residual(A, b, x) < tol:
        return x, k + 1, True
      rs_new = r @ r
      beta = rs_new / rs_old
      p = r + beta * p
      rs_old = rs_new
    return x, max_iter, False
