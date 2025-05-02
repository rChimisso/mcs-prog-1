# MatIter

##### *2025, Metodi del Calcolo Scientifico, Riccardo Chimisso 866009 & Mauro Zorzin 866001*

## Description

A simple project made for our university course *Metodi del Calcolo Scientifico* that provides iterative solvers for Symmetric Positive Definite (SPD) systems.

This project includes:

- **Documentation**: Extensive [Readme](/README.md), [Changelog](/CHANGELOG.md), and [Sphinx-generated codebase docs](https://rChimisso.github.io/mcs-prog-1/).
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
Documentation for the latest release is already live at [rChimisso.github.io/mcs-prog-1](https://rChimisso.github.io/mcs-prog-1/).  

The structured documentation can be generated with [Sphinx](https://www.sphinx-doc.org/en/master/).  
To build the documentation yourself, simply run the following command under the `docs/` directory:
```powershell
make html
```
To view it, simply open the file `docs/build/html/index.html` with a browser.

## Usage

TODO

To build the `MatIterEngine` executable yourself, simply run the following command in the project root:
```powershell
pyinstaller ./src/main.py --name MatIterEngine --noconsole --onefile
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
