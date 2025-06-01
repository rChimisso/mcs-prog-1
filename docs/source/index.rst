MatIter documentation
======================

Description
-----------

A simple project made for our university course *Metodi del Calcolo Scientifico* that provides iterative solvers for Symmetric Positive Definite (SPD) systems.

This project includes:

- **Documentation**: Extensive Readme, Changelog, and Sphinx-generated codebase docs.
- **Report**: A short report with results and analysis.
- **Test data**: A few data samples in `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_.
- **CI**: Automatic code analysis and deployment.
- **Releases**: Prebuilt `executables <https://github.com/rChimisso/mcs-prog-1/releases>`_ for Linux and Windows.

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

| **How it updates the solution**
|
| Split :math:`A` into its diagonal part :math:`D` and remainder :math:`R` so that :math:`A = D - R`. Because :math:`D` is diagonal its inverse is just the reciprocals of those diagonal numbers.
| One step is

.. math::

   x^{k+1} = D^{-1}(b - R x^{k})

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
| Here :math:`D` is the lower-triangular part of :math:`A` (diagonal included). 
| Instead of forming :math:`D^{-1}` we solve the triangular system :math:`D y = r` with forward substitution, then set :math:`x^{k+1}=x^{k}+y`.
|
| **Note**: in our implementation, the code appears to be different from what stated above, but it is **not** just a Jacobi in disguise.
| It splits :math:`A=D+L+U` (``np.diag``, ``np.tril``, ``np.triu``), then performs a forward sweep where each row update (``x[i] = (b[i] - L[i]@x - U[i]@x)/D[i]``) solves the :math:`i`-th equation of :math:`(D+L)\,x^{k+1}=b-Ux^{k}` using the freshly updated :math:`x^{k+1}_{0…i-1}` already stored in-place, while the still-old :math:`x^{k}_{i+1…n-1}` appear in the upper-part product — precisely the lower-triangular forward-substitution Gauss-Seidel requires and unlike Jacobi, which would recompute all entries at once from :math:`x^{k}`.  
| Dividing by the scalar :math:`D_{i}` avoids forming :math:`D^{-1}`, and for any symmetric positive-definite :math:`A` this iteration converges, so the loop stops as soon as the residual falls below `tol`.
| 
| **Why/when it converges**
| 
| The same strict diagonal dominance guarantees convergence.
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

   \phi(x) = \tfrac12 x^T A x - b^T x

With :math:`x^k` we take the steepest downhill direction :math:`r^k = b - A x^k` and an exact step length

.. math::

   \alpha^{k} = \frac{r^{k\,T}r^{k}}{r^{k\,T}A r^{k}}

| **Why/when it converges**
| 
| Because :math:`A` is SPD, :math:`\phi` is a bowl-shaped surface with one unique bottom point, so the iterations always reach the solution.
| Gradient descent, however, can *zig-zag* on a quadratic :math:`f(x)=\tfrac12x^\top Ax-b^\top x` when :math:`A` is ill-conditioned because each gradient :math:`Ax_k` is dominated by the steep, large-eigenvalue direction, so the step shoots almost perpendicular across the long, flat valley defined by the small eigenvalue; the next gradient then points back the other way, producing an alternating back-and-forth path that creeps only slowly along the valley floor—the worse the eigenvalue ratio :math:`\kappa(A)=\lambda_{\max}/\lambda_{\min}`, the sharper the zig-zag and the slower the convergence.
| 
| **Work per step**
| 
| One matrix-vector product + a few dot products → :math:`O(n^2)` for dense :math:`A`.

Conjugate Gradient Solver
~~~~~~~~~~~~~~~~~~~~~~~~~

**How it differs from Gradient Descent**

| CG keeps the same energy function :math:`\phi` but chooses each search direction :math:`p^k` so that it is *A-conjugate* to all previous ones (:math:`p^{i\,T}A p^{j}=0` if :math:`i\ne j`).
| A compact version of the update is:

.. math::

   \begin{align*}
   \alpha^k     & = \frac{r^{kT} \cdot r^k}{p^k \cdot A p^k} \\
   x^{k+1}      & = x^k + \alpha^k p^k \\
   r^{k+1}      & = r^k - \alpha^k A p^k \\
   \beta^k      & = \frac{r^{k+1} \cdot r^{k+1}}{r^{kT} \cdot r^k} \\
   p^{k+1}      & = r^{k+1} + \beta^k p^k
   \end{align*}

| Gradient Descent always moves *orthogonally* to level sets of :math:`\phi(x)=\tfrac12 x^T A x - b^T x`, so on an elongated quadratic its path keeps *bouncing* across the valley floor: steepest-descent directions at successive points are almost at right angles, giving the already mentioned *zig-zag*.  
| CG replaces these directions by **A-conjugate directions**.  
| Two non-zero vectors :math:`p_i,p_j` are A-conjugate if :math:`p_i^TA p_j=0`. In the metric induced by :math:`A` this means they point along **independent principal axes** of the quadratic; once you have minimised along one such axis you will never undo that progress while travelling along another. Hence the path bends only once per axis and quickly beelines to the minimiser.

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - **step**
     - **formula**
     - **geometric / algebraic role**
   * - :math:`\alpha^{k}`
     - :math:`\dfrac{r^{kT}\,r^{k}}{p^{k}\,A\,p^{k}}`
     - exact 1-D minimiser of :math:`\phi` along :math:`p^{k}` (*line search*)
   * - :math:`x^{k+1}`
     - :math:`x^{k}+\alpha^{k}p^{k}`
     - new iterate
   * - :math:`r^{k+1}`
     - :math:`r^{k}-\alpha^{k}A\,p^{k}`
     - new residual; equal to :math:`-\nabla\!\phi\!\left(x^{k+1}\right)` and
       :math:`A`-orthogonal to :math:`p^{k}`
   * - :math:`\beta^{k}`
     - :math:`\dfrac{r^{\,k+1}\!\cdot r^{\,k+1}}{r^{kT}\,r^{k}}`
     - scale factor that ensures :math:`p^{k+1}` is :math:`A`-conjugate to
       :math:`p^{k}`
   * - :math:`p^{k+1}`
     - :math:`r^{k+1}+\beta^{k}p^{k}`
     - new search direction that blends fresh gradient information with the
       past direction just enough to maintain :math:`A`-conjugacy

| The inner-product ratios arise from the algebraic condition :math:`p^{(k+1)T}A p^k = 0`, solved for :math:`\beta^k` using :math:`r^{k+1} = r^k - \alpha^kA p^k`.
|
| **Why/when it converges**
| 
| For an :math:`n × n` SPD matrix :math:`A`, there are surely at least :math:`n` residuals :math:`\{r^0,…,r^{n-1}\}`. Because each new direction is A-conjugate to all previous ones, these :math:`n` directions form a basis of :math:`ℝ^n`; after at most :math:`n` iterations the algorithm has minimised :math:`\phi` along every independent axis and lands exactly at the unique solution :math:`x`.
| In exact arithmetic this is a proof; in floating point, rounding blurs conjugacy and we need :math:`≈√κ(A)` iterations, still much faster than plain GD.
| 
| **Work per step**
| 
| One matrix-vector product plus a fixed number of vector operations, again about :math:`O(n^2)` but with many fewer iterations overall.

Contents
--------

.. toctree::
   :maxdepth: 2

   solvers
