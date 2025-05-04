from typing import Any, Final, cast
from abc import ABC, abstractmethod
import pytest
import scipy.io # type: ignore
import numpy as np
from solvers import IterativeSolver, JacobiSolver, GaussSeidelSolver, GradientDescentSolver, ConjugateGradientSolver

class AbstractIterativeSolverTest(ABC):
  @abstractmethod
  def provide(self) -> IterativeSolver:
    pass

  def test_solve(self):
    solver: Final[IterativeSolver] = self.provide()
    for data in ["spa1", "spa2", "vem1", "vem2"]:
      with open(f"./data/{data}.mtx", encoding="ascii") as mtx:
        A = cast(np.typing.NDArray[Any], scipy.io.mmread(mtx).toarray()) # type: ignore
        x = np.ones(A.shape[1])
        b = A @ x
        x_apx, info = solver.solve(A, b)
        rel_err = np.linalg.norm(x_apx - x) / np.linalg.norm(x)
        assert np.allclose(x_apx, x, atol=5e-2)
        assert rel_err < 1e-3
        assert info.converged
        assert info.iterations < 20_000
        assert info.residual < 1e-6

class TestJacobiSolver(AbstractIterativeSolverTest):
  def provide(self) -> IterativeSolver:
    return JacobiSolver(1e-6)

class TestGaussSeidelSolver(AbstractIterativeSolverTest):
  def provide(self) -> IterativeSolver:
    return GaussSeidelSolver(1e-6)

class TestGradientDescentSolver(AbstractIterativeSolverTest):
  def provide(self) -> IterativeSolver:
    return GradientDescentSolver(1e-6)

class TestConjugateGradientSolver(AbstractIterativeSolverTest):
  def provide(self) -> IterativeSolver:
    return ConjugateGradientSolver(1e-6)

if __name__ == "__main__":
  pytest.main()
