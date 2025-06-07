# MatIter

##### *2025, Metodi del Calcolo Scientifico, Riccardo Chimisso 866009 & Mauro Zorzin 866001*

## Description

A simple project made for our university course *Metodi del Calcolo Scientifico* that provides iterative solvers for Symmetric Positive Definite (SPD) systems.

This project includes:

- **Documentation**: Extensive [Readme](/README.md), [Changelog](/CHANGELOG.md), and [Sphinx-generated codebase docs](https://rchimisso.github.io/mcs-prog-1/).
- **Report**: A [short report](/REPORT.md) with results and analysis.
- **Test data**: A few [data samples](/data/) in [Matrix Market format](https://math.nist.gov/MatrixMarket/formats.html).
- **CI**: Automatic code analysis and deployment.
- **Releases**: Prebuilt [executables](https://github.com/rChimisso/mcs-prog-1/releases) for Linux and Windows.

## Setup

Setting up the environment is pretty easy:

1. Set up **Python 3.12.9** (you can use any environment manager or none).
2. Install the dependencies from the file [`requirements.txt`](/requirements.txt).

The suggested IDE is [Visual Studio Code](https://code.visualstudio.com/), and settings for it are included.

## Documentation

The source code is fully documented with Docstrings in [reST](https://docutils.sourceforge.io/rst.html).  
Documentation for the latest release is already live at [rChimisso.github.io/mcs-prog-1](https://rchimisso.github.io/mcs-prog-1/).  

The structured documentation can be generated with [Sphinx](https://www.sphinx-doc.org/en/master/).  
To build the documentation yourself, simply run the following command under the `docs/` directory:
```powershell
make html
```
To view it, simply open the file `docs/build/html/index.html` with a browser.

## Usage

```
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
```

You can either use the prebuilt [executables](https://github.com/rChimisso/mcs-prog-1/releases) for your platform, or build it yourself.

To build the `MatIterEngine` executable yourself, simply run the following command in the project root:
```powershell
pyinstaller ./src/main.py --name MatIterEngine --onefile
```
This will create an executable for your platform.

## Solvers

This project provides 4 iterative solvers for Symmetric Positive Definite (SPD) linear systems.

> **What “SPD” means**
> A matrix is *symmetric positive definite* (SPD) when it equals its transpose and all its eigen‑values are positive. In that case every quadratic form $x^T A x$ is strictly positive unless $x=0$.

---

### Jacobi Solver

#### How it updates the solution

Split $A$ into its diagonal part $D$ and remainder $R$ so that $A = D + R$. Because $D$ is diagonal its inverse is just the reciprocals of those diagonal numbers.  
One step is

$x^{k+1} = D^{-1}(b - R x^{k})$

#### Why/when it converges

If every row has a *strictly dominating* diagonal entry – that is, $|a_{ii}|$ is larger than the sum of the other absolute values in that row – Jacobi certainly converges.  
Many SPD matrices meet (or can be scaled to meet) this requirement, so in practice Jacobi often works for SPD systems.

#### Work per step

One matrix–vector product plus a few vector operations → about $O(n^2)$ operations for a dense matrix.

---

### Gauss–Seidel Solver

#### How it differs from Jacobi

Here $D$ is the lower‑triangular part of $A$ (diagonal included).  
Instead of forming $D^{-1}$ we solve the triangular system $Dy = r$ with forward substitution, then set $x^{k+1}=x^{k}+y$.

**Note**: in our implementation, the code appears to be different from what stated above, but it is **not** just a Jacobi in disguise.  
It splits $A=D+L+U$ (`np.diag`, `np.tril`, `np.triu`), then performs a forward sweep where each row update (`x[i] = (b[i] - L[i]@x - U[i]@x)/D[i]`) solves the $i$-th equation of $(D+L)x^{k+1}=b-Ux^{k}$ using the freshly updated $x^{k+1}\_{0…i-1}$ already stored in-place, while the still-old $x^{k}\_{i+1…n-1}$ appear in the upper-part product — precisely the lower-triangular forward-substitution Gauss-Seidel requires and unlike Jacobi, which would recompute all entries at once from $x^{k}$.  
Dividing by the scalar $D_{ii}$ avoids forming $D^{-1}$, and for any symmetric positive-definite $A$ this iteration converges, so the loop stops as soon as the residual falls below `tol`.

#### Why/when it converges

The same strict diagonal dominance guarantees convergence.  

#### Work per step

Still dominated by one matrix–vector product, so again $O(n^2)$ for dense $A$, but usually needs fewer sweeps than Jacobi.

---

### Gradient Descent Solver

#### Geometric picture

Solving $Ax = b$ is the same as minimising the quadratic

$\phi(x)=\tfrac12 x^T Ax - b^T x$

With $x^k$ we take the steepest downhill direction $r^k=b – Ax^k$ and an exact step length

$\alpha^{k}=\frac{r^{k\,T}r^{k}}{r^{k\,T}A r^{k}}$

#### Why/when it converges

Because $A$ is SPD, $\phi$ is a bowl-shaped surface with one unique bottom point, so the iterations always reach the solution.  
Gradient descent, however, can *zig-zag* on a quadratic $f(x)=\tfrac12x^\top Ax-b^\top x$ when $A$ is ill-conditioned because each gradient $Ax_k$ is dominated by the steep, large-eigenvalue direction, so the step shoots almost perpendicular across the long, flat valley defined by the small eigenvalue; the next gradient then points back the other way, producing an alternating back-and-forth path that creeps only slowly along the valley floor—the worse the eigenvalue ratio $\kappa(A)=\lambda_{\max}/\lambda_{\min}$, the sharper the zig-zag and the slower the convergence.

#### Work per step

One matrix–vector product + a few dot products → $O(n^2)$ for dense $A$.

---

### Conjugate Gradient Solver

#### How it differs from Gradient Descent

CG keeps the same energy function $\phi$ but chooses each search direction $p^k$ so that it is *A‑conjugate* to all previous ones ($p^{i\,T}A p^{j}=0$ if i≠j).  
A compact version of the update is:

1. $α^k = \frac{r^{kT} \cdot r^k}{p^k \cdot A p^k}$

2. $x^{k+1} = x^k+α^kp^k$

3. $r^{k+1} = r^k−α^kA p^k$

4. $\beta^k = \frac{r^{k+1} \cdot r^{k+1}}{r^{kT} \cdot r^k}$

5. $p^{k+1} = r^{k+1}+\beta^kp^k$

Gradient Descent always moves *orthogonally* to level sets of $\phi(x)=\tfrac12 x^T A x - b^T x$, so on an elongated quadratic its path keeps *bouncing* across the valley floor: steepest-descent directions at successive points are almost at right angles, giving the already mentioned *zig-zag*.  
CG replaces these directions by **A-conjugate directions**.  
Two non-zero vectors $p_i,p_j$ are A-conjugate if $p_i^TA p_j=0$. In the metric induced by $A$ this means they point along **independent principal axes** of the quadratic; once you have minimised along one such axis you will never undo that progress while travelling along another. Hence the path bends only once per axis and quickly beelines to the minimiser.

| step      | formula                                          | geometric / algebraic role                                                                                   |
| --------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| $α^k$     | $\frac{r^{kT} \cdot r^k}{p^k \cdot A p^k}$       | exact $1D$ minimiser of $\phi$ along $p^k$ (line search)                                                     |
| $x^{k+1}$ | $x^k+α^kp^k$                                     | new iterate                                                                                                  |
| $r^{k+1}$ | $r^k−α^kA p^k$                                   | new residual; equal to $−\nabla\phi(x^{k+1})$ and A-orthogonal to $p^k$                                      |
| $\beta^k$ | $\frac{r^{k+1} \cdot r^{k+1}}{r^{kT} \cdot r^k}$ | scale factor that ensures $p^{k+1}$ is A-conjugate to $p^k$                                                  |
| $p^{k+1}$ | $r^{k+1}+\beta^kp^k$                             | new search direction that blends fresh gradient info with past direction just enough to maintain A-conjugacy |

The inner-product ratios arise from the algebraic condition $p^{(k+1)T}A p^k = 0$, solved for $\beta^k$ using $r^{k+1} = r^k - \alpha^kA p^k$.

#### Why/when it converges

For an $n × n$ SPD matrix $A$, there are surely at least $n$ residuals $\{r^0,…,r^{n-1}\}$. Because each new direction is A-conjugate to all previous ones, these $n$ directions form a basis of $ℝ^n$; after at most $n$ iterations the algorithm has minimised $\phi$ along every independent axis and lands exactly at the unique solution $x$.
In exact arithmetic this is a proof; in floating point, rounding blurs conjugacy and we need $≈√κ(A)$ iterations, still much faster than plain GD.

#### Work per step

One matrix–vector product plus a fixed number of vector operations, again about $O(n^2)$ but with many fewer iterations overall.
