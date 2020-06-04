# LSSPAR
>A Python Package to solve sparsity constrained least squares problems

This repository contains a package to solve the following problem

[![g](https://github.com/Fatih-S-AKTAS/LSSPAR/blob/master/files/lssparquestion.png)]()

> Assumptions
- System is overdetermined, A is a m x n matrix where m > n



# Example Usage

This package uses guppy3 by  YiFei Zhu and Sverker Nilsson for tracking memory usage. Hence guppy3 must be installed prior to using LSSPAR.

```python
pip install guppy3
```

Then, after downloading LSSPAR.py, it can be used as follows:

```python
from LS_SPAR import * 

A = Data Matrix
b = 
s = sparsity level

question = LSSPAR(A,b,s)
```
