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

This project provides 4 different iterative solvers.

### Jacobi Solver

TODO

### Gauss-Seidel Solver

TODO

### Gradient Descent Solver

TODO

### Conjugate Gradient Solver

TODO
