from typing import Any, Final, cast
import scipy.io
import numpy as np
from solvers import IterativeSolver, JacobiSolver, GaussSeidelSolver, GradientDescentSolver, ConjugateGradientSolver

VERSION: Final[str] = "0.0.1"

def solve(solver: IterativeSolver, name: str, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x: np.typing.NDArray[Any], tol: float) -> None:
  print(f"{name} Solver")
  x_apx, info = solver.solve(A, b, tol=tol)
  rel_err = np.linalg.norm(x_apx - x) / np.linalg.norm(x)
  print(f"  Rel. error:\t{rel_err}")
  print(f"  Iterations:\t{info.iterations}")
  print(f"  Time elapsed:\t{info.time}")
  print()

def main() -> None:
  print(f"MatIter iterative solver engine v{VERSION}\n")

  # Take A, b, x, and tol as inputs from command line.
  # For each solver:
  #   solve
  #   print rel_err, num_iter, time

  with open("./data/vem1.mtx", encoding="ascii") as mtx:
    A = cast(np.typing.NDArray[Any], scipy.io.mmread(mtx).toarray())
    x = np.ones(A.shape[1])
    b = A @ x

    solve(JacobiSolver(), "Jacobi", A, b, x, 1e-10)
    solve(GaussSeidelSolver(), "Gauss Seidel", A, b, x, 1e-10)
    solve(GradientDescentSolver(), "Gradient Descent", A, b, x, 1e-10)
    solve(ConjugateGradientSolver(), "Conjugate Gradient", A, b, x, 1e-10)

if __name__ == "__main__":
  main()
