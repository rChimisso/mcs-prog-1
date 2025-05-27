MatIter documentation
======================

Description
-----------

| A simple project made for our university course *Metodi del Calcolo Scientifico* that provides iterative solvers for Symmetric Positive Definite (SPD) systems.
| 
| This project includes:
| 
| - **Documentation**: Extensive Readme, Changelog, and Sphinx-generated codebase docs.
| - **Report**: A short report with results and analysis.
| - **Test data**: A few data samples in `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_.
| - **CI**: Automatic code analysis and deployment.
| - **Releases**: Prebuilt `executables <https://github.com/rChimisso/mcs-prog-1/releases>`_ for Linux and Windows.

Setup
-----

Setting up the environment is pretty easy:

1. Set up **Python 3.12.9** (you can use any environment manager or none).
2. Install the dependencies from the file ``requirements.txt``.

The suggested IDE is `Visual Studio Code <https://code.visualstudio.com/>`_, and settings for it are included.

Usage
-----

.. code::

   usage: MatIterEngine [-h] [-A A_POS] [-x X_POS] [-tol TOL_POS] A [x] [tol]
   
   Compares four different iterative solvers (Jacobi, Gauss-Seidel, Gradient Descent, Conjugate Gradient) in solving the SPD system defined by Ax = b.
   
   positional arguments:
     A                     Path to the .mtx file for the A matrix.
     x                     Path to the .mtx file for the exact solution vector x; default [1., ..., 1.].
     tol                   Floating-point tolerance (≥ 1e-14); default 1e-08.
   
   options:
     -h, --help            show this help message and exit
     -A A_POS, --A A_POS, -matrix A_POS, --matrix A_POS
                           Named form of the A-matrix argument. Overrides the positional A if both are given.
     -x X_POS, --x X_POS, -sol X_POS, --sol X_POS
                           Named form of the solution vector x; overrides the positional x if both are given.
     -tol TOL_POS, --tol TOL_POS
                           Named form of the tolerance tol; overrides the positional tol if both are given.

You can either use the prebuilt `executables <https://github.com/rChimisso/mcs-prog-1/releases>`_ for your platform, or build it yourself.

To build the ``MatIterEngine`` executable yourself, simply run the following command in the project root:

.. code:: powershell

   pyinstaller ./src/main.py --name MatIterEngine --onefile

This will create an executable for your platform.

Solvers
-------

This project provides 4 iterative solvers for Symmetric Positive Definite (SPD) linear systems.

.. note::
   **What “SPD” means**

   A matrix is *symmetric positive definite* (SPD) when it equals its transpose and all its eigen-values are positive. In that case every quadratic form :math:`x^T A x` is strictly positive unless :math:`x=0`.

Jacobi Solver
~~~~~~~~~~~~~

| **How it updates the guess**
|
| Split :math:`A` into its diagonal part :math:`P` and everything else :math:`N` so that :math:`A = P - N`. Because :math:`P` is diagonal its inverse is just the reciprocals of those diagonal numbers.
| One step is

.. math::

   x^{k+1} = P^{-1}(N x^{k} + b)

| **Why/when it converges**
|
| If every row has a *strictly dominating* diagonal entry - that is, :math:`|a_{ii}|` is larger than the sum of the other absolute values in that row - Jacobi certainly converges.
| Many SPD matrices meet (or can be scaled to meet) this requirement, so in practice Jacobi often works for SPD systems.
|
| **Work per step**
|
| One matrix-vector product plus a few vector operations → about :math:`O(n^2)` operations for a dense matrix.

Gauss-Seidel Solver
~~~~~~~~~~~~~~~~~~~

| **How it differs from Jacobi**
| 
| Here :math:`P` is the lower-triangular part of :math:`A` (diagonal included). 
| Instead of forming :math:`P^{-1}` we solve the triangular system :math:`P y = r` with forward substitution, then set :math:`x^{k+1}=x^{k}+y`.
| 
| **Why/when it converges**
| 
| The same strict diagonal dominance guarantees convergence.
| A popular variant called SOR adds a relaxation factor :math:`\omega`; for SPD matrices it converges as long as :math:`0 < \omega < 2`.
| 
| **Work per step**
| 
| Still dominated by one matrix-vector product, so again :math:`O(n^2)` for dense :math:`A`, but usually needs fewer sweeps than Jacobi.

Gradient Descent Solver
~~~~~~~~~~~~~~~~~~~~~~~

| **Geometric picture**
| 
| Solving :math:`A x = b` is the same as minimising the quadratic

.. math::

   \varphi(x) = \tfrac12 x^T A x - b^T x

With :math:`x^k` we take the steepest downhill direction :math:`r^k = b - A x^k` and an exact step length

.. math::

   \alpha^{k} = \frac{r^{k\,T}r^{k}}{r^{k\,T}A r^{k}}

| **Why/when it converges**
| 
| Because :math:`A` is SPD, :math:`\phi` is a bowl-shaped surface with one unique bottom point, so the iterations always reach the solution.
| The path, however, can *zig-zag* when :math:`A`'s eigen-values are far apart.
| 
| **Work per step**
| 
| One matrix-vector product + a few dot products → :math:`O(n^2)` for dense :math:`A`.

Conjugate Gradient Solver
~~~~~~~~~~~~~~~~~~~~~~~~~

**Key idea**

| CG keeps the same energy function :math:`\phi` but chooses each search direction :math:`p^k` so that it is *A-conjugate* to all previous ones (:math:`p^{i\,T}A p^{j}=0` if :math:`i\ne j`).
| With this choice the exact solution is obtained in at most :math:`n` steps in exact arithmetic.
| 
| A compact version of the update is:

.. math::

   \begin{align*}
   \alpha^k     & = \frac{r^k \cdot r^k}{p^k \cdot A p^k} \\
   x^{k+1}      & = x^k + \alpha^k p^k \\
   r^{k+1}      & = r^k - \alpha^k A p^k \\
   \beta^k      & = \frac{r^{k+1} \cdot r^{k+1}}{r^k \cdot r^k} \\
   p^{k+1}      & = r^{k+1} + \beta^k p^k
   \end{align*}

| **Why/when it converges**
| 
| CG needs :math:`A` to be SPD.
| In floating-point arithmetic it usually reaches machine precision in roughly :math:`\sqrt{\kappa(A)}` iterations, far faster than plain Gradient Descent.
| 
| **Work per step**
| 
| One matrix-vector product plus a fixed number of vector operations, again about :math:`O(n^2)` for dense :math:`A` but with many fewer iterations overall.

Contents
--------

.. toctree::
   :maxdepth: 2

   solvers
