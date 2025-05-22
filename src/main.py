import sys
import math
import argparse
from pathlib import Path
from typing import Any, Final, Optional
import scipy.io # type: ignore
import scipy.sparse # type: ignore
import numpy as np
from solvers import IterativeSolver, JacobiSolver, GaussSeidelSolver, GradientDescentSolver, ConjugateGradientSolver

VERSION: Final[str] = "0.0.1"

def validate_inputs(A: str, x: Optional[str], tol: float) -> list[str]:
  """
  Validates the script inputs.

  :param A: Path to the .mtx file for the A matrix.
  :type A: str
  :param x: Path to the .mtx file for the x vector.
  :type x: Optional[str]
  :param tol: Tolerance value.
  :type tol: float
  :return: List of errors.
  :rtype: list[str]
  """
  errors: list[str] = []
  if not Path(A).is_file():
    errors.append(f"Error: file '{A}' does not exist.")
  if x is not None and not Path(x).is_file():
    errors.append(f"Error: file '{x}' does not exist.")
  if math.isnan(tol) or not math.isfinite(tol) or tol < 1e-14:
    errors.append("Error: 'tol' must be a finite float ≥ 1e-14.")
  return errors

def as_ndarray(file: str) -> np.typing.NDArray[Any]:
  """
  Converts the data contained in the specified .mtx file into a numpy array.

  :param file: Path to the .mtx file.
  :type file: str
  :return: numpy array.
  :rtype: np.typing.NDArray[Any]
  """
  with open(file, encoding="ascii") as mtx:
    matrix = scipy.io.mmread(mtx) # type: ignore
    return matrix.toarray() if scipy.sparse.issparse(matrix) else np.asarray(matrix) # type: ignore

def solve(solver: IterativeSolver, name: str, A: np.typing.NDArray[Any], b: np.typing.NDArray[Any], x: np.typing.NDArray[Any], tol: float) -> None:
  """
  | Uses the given solver to solve the system defined by A, b, and x.
  | Prints relevant data at the end.

  :param solver: Solver to use.
  :type solver: IterativeSolver
  :param name: Name of the solver.
  :type name: str
  :param A: A matrix.
  :type A: np.typing.NDArray[Any]
  :param b: b vector.
  :type b: np.typing.NDArray[Any]
  :param x: exact solution vector.
  :type x: np.typing.NDArray[Any]
  :param tol: Tolerance.
  :type tol: float
  """
  print(f"{name} Solver")
  x_apx, info = solver.solve(A, b, tol=tol)
  rel_err = np.linalg.norm(x_apx - x) / np.linalg.norm(x)
  print(f"  Rel. error:\t{rel_err}")
  print(f"  Iterations:\t{info.iterations}")
  print(f"  Time elapsed:\t{info.time} s")
  print()

def main() -> None:
  """
  Main script.

  Prints out the engine header and then solves the system specified by the input arguments with all available iterative solvers.
  """
  print(f"MatIterEngine v{VERSION} - Iterative solvers engine for SPD systems\n")

  parser = argparse.ArgumentParser(
    prog="MatIterEngine",
    description="Compares four different iterative solvers (Jacobi, Gauss-Seidel, Gradient Descent, Conjugate Gradient) in solving the SPD system defined by Ax = b."
  )
  parser.add_argument("A", type=str, metavar="A", help="Path to the .mtx file for the A matrix.")
  parser.add_argument("-A", "--A", "-matrix", "--matrix", dest="A_pos", type=str, help="Named form of the A-matrix argument. Overrides the positional A if both are given.")
  parser.add_argument("x", type=str, metavar="x", nargs="?", default=None, help="Path to the .mtx file for the exact solution vector x; default [1., ..., 1.].")
  parser.add_argument("-x", "--x", "-sol", "--sol", dest="x_pos", type=str, default=None, help="Named form of the solution vector x; overrides the positional x if both are given.")
  parser.add_argument("tol", type=float, nargs="?", default=1e-8, help=f"Floating-point tolerance (≥ {1e-14}); default {1e-8}.")
  parser.add_argument("-tol", "--tol", dest="tol_pos", type=float, default=None, help="Named form of the tolerance tol; overrides the positional tol if both are given.")
  args = parser.parse_intermixed_args()

  args.A = args.A_pos or args.A
  args.x = args.x_pos or args.x
  args.tol = args.tol_pos or args.tol

  errors = validate_inputs(args.A, args.x, args.tol)
  if errors:
    for message in errors:
      print(message, file=sys.stderr)
    sys.exit(1)

  A = as_ndarray(args.A)
  x = as_ndarray(args.x)[:, 0] if args.x else np.ones(A.shape[1])
  b = A @ x

  solve(JacobiSolver(), "Jacobi", A, b, x, args.tol)
  solve(GaussSeidelSolver(), "Gauss-Seidel", A, b, x, args.tol)
  solve(GradientDescentSolver(), "Gradient Descent", A, b, x, args.tol)
  solve(ConjugateGradientSolver(), "Conjugate Gradient", A, b, x, args.tol)

if __name__ == "__main__":
  main()
