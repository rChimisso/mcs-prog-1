import time
from typing import Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

bar_format = "  {desc}: {percentage:.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}{postfix}]"

@dataclass
class Info:
  """
  Solver Info data class.
  """
  converged: bool
  """
  Whether the method has converged.
  """
  iterations: int
  """
  Iterations performed.
  """
  residual: float
  """
  Residual error.
  """
  time: float
  """
  Time elapsed.
  """

class IterativeSolver(ABC):
  """
  Method agnostic iterative solver for SPD matrices.
  """
  def __init__(self, tol: float = 1e-8, max_iter: int = 20_000) -> None:
    """
    Method agnostic iterative solver for SPD matrices.

    :param tol: Fallback for the residual error tolerance to deem a solution acceptable, defaults to 1e-8.
    :type tol: float, optional
    :param max_iter: Fallback for the maximum amount of iterations to perform, defaults to 20000.
    :type max_iter: int, optional
    """
    super().__init__()
    self.tol: float = tol
    self.max_iter: int = max_iter

  def solve(self, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x0: Optional[np.typing.NDArray[Any]] = None, tol: Optional[float] = None, max_iter: Optional[int] = None) -> tuple[np.typing.NDArray[Any], Info]:
    """
    Iteratively solves the system Ab = x, starting from x0, using the provided tolerance and maximum amount of iterations.

    :param A: System matrix.
    :type A: np.typing.NDArray[Any]
    :param b: b vector.
    :type b: np.typing.NDArray[Any]
    :param x0: Starting solution, defaults to all zeros.
    :type x0: Optional[np.typing.NDArray[Any]], optional
    :param tol: Residual error tolerance, defaults to this solver's tolerance.
    :type tol: Optional[float], optional
    :param max_iter: Maximum amount of iterations, defaults to this solver's maximum amount of iterations.
    :type max_iter: Optional[int], optional
    :return: A solution for the system (if converged), and infos on the run.
    :rtype: tuple[np.typing.NDArray[Any], Info]
    """
    t0 = time.perf_counter()
    x, iterations, converged = self._solve(A.astype(np.float64), b.astype(np.float64), (x0 or np.zeros_like(b)).astype(np.float64), self.tol if tol is None else tol, self.max_iter if max_iter is None else max_iter)
    info = Info(converged, iterations, self._residual(A, b, x), time.perf_counter() - t0)
    return x, info

  def _residual(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x: np.typing.NDArray[np.float64]) -> float:
    """
    Returns the relative residual error defined as *|| Ax - b || / || b ||*.

    :param A: System matrix.
    :type A: np.typing.NDArray[Any]
    :param b: b vector.
    :type b: np.typing.NDArray[Any]
    :param x: Approximate solution.
    :type x: np.typing.NDArray[Any]
    :return: Relative residual error.
    :rtype: float
    """
    return (np.linalg.norm(A @ x - b) / np.linalg.norm(b)).item()

  @abstractmethod
  def _solve(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x0: np.typing.NDArray[np.float64], tol: float, max_iter: int) -> tuple[np.typing.NDArray[np.float64], int, bool]:
    """
    Internal solver method.

    :param A: System matrix.
    :type A: np.typing.NDArray[np.float64]
    :param b: b vector.
    :type b: np.typing.NDArray[np.float64]
    :param x0: Starting solution.
    :type x0: np.typing.NDArray[np.float64]
    :param tol: Residual error tolerance.
    :type tol: float
    :param max_iter: Maximum amount of iterations.
    :type max_iter: int
    :return: The approximate solution, the amount of iterations performed, and whether the method converged.
    :rtype: tuple[np.typing.NDArray[np.float64], int, bool]
    """
    raise NotImplementedError

class JacobiSolver(IterativeSolver):
  """
  Jacobi solver.

  Uses Jacobi's method to iteratively solve a SPD system.  
  Splits **A** into its diagonal **D** and remainder **R = A - D**.  
  Starting from an initial guess *x⁰*, each forward sweep computes *xᵏ⁺¹ ← D⁻¹(b - R xᵏ)*, so every component is updated simultaneously using only the values from the previous iteration.  
  Jacobi converges when the iteration matrix *D⁻¹R* has spectral radius *< 1* (e.g., for strictly diagonally-dominant or suitably scaled SPD systems).
  """
  def _solve(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x0: np.typing.NDArray[np.float64], tol: float, max_iter: int) -> tuple[np.typing.NDArray[np.float64], int, bool]:
    x = x0
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in tqdm(range(max_iter), bar_format=bar_format, ncols=80, unit=" iter"):
      x = (b - R @ x) / D
      if self._residual(A, b, x) < tol:
        return x, k + 1, True
    return x, max_iter, False

class GaussSeidelSolver(IterativeSolver):
  """
  Gauss-Seidel solver.

  Uses Gauss-Seidel's method to iteratively solve a SPD system.  
  Splits **A** into its diagonal **D**, strictly lower **L**, and strictly upper **U**.  
  Each forward sweep updates entries in-place: *for i = 0,…,n-1: xᵢ ← (bᵢ - Lᵢx - Uᵢx) / Dᵢ*.  
  Gauss-Seidel is guaranteed to converge for np.float64 SPD matrix eventually.
  """
  def _solve(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x0: np.typing.NDArray[np.float64], tol: float, max_iter: int) -> tuple[np.typing.NDArray[np.float64], int, bool]:
    x, n = x0, b.size
    D, L, U = np.diag(A), np.tril(A, -1), np.triu(A,  1)
    for k in tqdm(range(max_iter), bar_format=bar_format, ncols=80, unit=" iter"):
      for i in range(n):
        x[i] = (b[i] - L[i] @ x - U[i] @ x) / D[i]
      if self._residual(A, b, x) < tol:
        return x, k + 1, True
    return x, max_iter, False

class GradientDescentSolver(IterativeSolver):
  """
  Gradient Descent solver.

  Uses the Gradient Descent method to iteratively solve a SPD system.  
  Minimises the quadratic *½xᵀAx - bᵀx* for SPD **A** by moving along the negative gradient (residual) with an exact line-search each step:  
  *rᵏ = b - A xᵏ*  
  *αᵏ = (rᵏᵀ rᵏ) / (rᵏᵀ A rᵏ)*  
  *xᵏ⁺¹ = xᵏ + αᵏ rᵏ*  
  Converges for every SPD system, with rate governed by the condition number *κ(A)*.
  """
  def _solve(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x0: np.typing.NDArray[np.float64], tol: float, max_iter: int) -> tuple[np.typing.NDArray[np.float64], int, bool]:
    x, r = x0, b - A @ x0
    if np.linalg.norm(r) > tol * np.linalg.norm(b):
      for k in tqdm(range(max_iter), bar_format=bar_format, ncols=80, unit=" iter"):
        Ar = A @ r
        alpha = (r @ r) / (r @ Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        if self._residual(A, b, x) < tol:
          return x, k + 1, True
      return x, max_iter, False
    return x, 0, True

class ConjugateGradientSolver(IterativeSolver):
  """
  Conjugate Gradient solver.

  Uses Conjugate Descent method to iteratively solve a SPD system.  
  Krylov-subspace method tailored to SPD systems; builds mutually **A-conjugate** search directions to minimise the A-norm of the error.  
  Starting with *r⁰ = b - A x⁰* and *p⁰ = r⁰*, repeat:  
  *αᵏ = (rᵏᵀ rᵏ) / (pᵏᵀ A pᵏ)*  
  *xᵏ⁺¹ = xᵏ + αᵏ pᵏ*  
  *rᵏ⁺¹ = rᵏ - αᵏ A pᵏ*  
  *βᵏ = (rᵏ⁺¹ᵀ rᵏ⁺¹) / (rᵏᵀ rᵏ)*  
  *pᵏ⁺¹ = rᵏ⁺¹ + βᵏ pᵏ*  
  Converges in *≤ n* steps in exact arithmetic and typically in *O(√κ(A))* iterations.
  """
  def _solve(self, A: np.typing.NDArray[np.float64], b: np.typing.NDArray[np.float64], x0: np.typing.NDArray[np.float64], tol: float, max_iter: int) -> tuple[np.typing.NDArray[np.float64], int, bool]:
    x, r = x0, b - A @ x0
    p, rs_old = r.copy(), r @ r
    for k in tqdm(range(max_iter), bar_format=bar_format, ncols=80, unit=" iter"):
      Ap = A @ p
      alpha = rs_old / (p @ Ap)
      x = x + alpha * p
      r = r - alpha * Ap
      if self._residual(A, b, x) < tol:
        return x, k + 1, True
      rs_new = r @ r
      try:
        beta = rs_new / rs_old
      except:
        print(rs_new, rs_old)
        beta = 0
      p = r + beta * p
      rs_old = rs_new
    return x, max_iter, False
