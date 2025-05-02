import scipy.io
import numpy as np
from typing import Any
from solvers import JacobiSolver, GaussSeidelSolver, GradientDescentSolver, ConjugateGradientSolver

VERSION = "0.0.1"

def main():
  print(f"MatIter iterative solver engine v{VERSION}\n")

  with open("./data/vem1.mtx") as mtx:
    A: np.typing.NDArray[Any] = scipy.io.mmread(mtx).toarray()
    x = np.ones(A.shape[1])
    b = A @ x

    jacobi_solver = JacobiSolver()
    gauss_seidel_solver = GaussSeidelSolver()
    gradient_descent_solver = GradientDescentSolver()
    conjugate_gradient_solver = ConjugateGradientSolver()

    jacobi_x, jacobi_info = jacobi_solver.solve(A, b)
    gauss_seidel_x, gauss_seidel_info = gauss_seidel_solver.solve(A, b)
    gradient_descent_x, gradient_descent_info = gradient_descent_solver.solve(A, b)
    conjugate_gradient_x, conjugate_gradient_info = conjugate_gradient_solver.solve(A, b)

    jacobi_rel_err = np.linalg.norm(jacobi_x - x) / np.linalg.norm(x)
    gauss_seidel_rel_err = np.linalg.norm(gauss_seidel_x - x) / np.linalg.norm(x)
    gradient_descent_rel_err = np.linalg.norm(gradient_descent_x - x) / np.linalg.norm(x)
    conjugate_gradient_rel_err = np.linalg.norm(conjugate_gradient_x - x) / np.linalg.norm(x)

    print()
    print("Jacobi Method")
    print(f"Relative error: {jacobi_rel_err}")
    print(f"Solution: {jacobi_x}")
    print(f"Infos: {jacobi_info}")
    print("--------------------------")
    print("Gauss Seidel Method")
    print(f"Relative error: {gauss_seidel_rel_err}")
    print(f"Solution: {gauss_seidel_x}")
    print(f"Infos: {gauss_seidel_info}")
    print("--------------------------")
    print("Gradient Descent Method")
    print(f"Relative error: {gradient_descent_rel_err}")
    print(f"Solution: {gradient_descent_x}")
    print(f"Infos: {gradient_descent_info}")
    print("--------------------------")
    print("Conjugate Gradient Method")
    print(f"Relative error: {conjugate_gradient_rel_err}")
    print(f"Solution: {conjugate_gradient_x}")
    print(f"Infos: {conjugate_gradient_info}")
    print()

if __name__ == "__main__":
  main()
