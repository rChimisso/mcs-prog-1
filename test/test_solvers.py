import pytest
import numpy as np
from solvers import JacobiSolver

class TestJacobiSolver:
  def test_solve(self):
    solver = JacobiSolver()
    A = np.eye(3, dtype=float)
    b = np.array([1.0, 2.0, 3.0])
    x, info = solver.solve(A, b)
    assert np.allclose(x, b)
    assert info is not None

if __name__ == "__main__":
  pytest.main()
