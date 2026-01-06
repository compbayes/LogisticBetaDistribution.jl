# LogisticBetaDistribution



[![Dev](https://img.shields.io/badge/docs-stable-blue.svg)](https://compbayes.github.io/LogisticBetaDistribution.jl/dev/)
[![Build Status](https://github.com/compbayes/LogisticBetaDistribution.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/compbayes/LogisticBetaDistribution.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/compbayes/LogisticBetaDistribution.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/compbayes/LogisticBetaDistribution.jl)

## Description

Julia implementation of the Logistic Beta distribution, also called [Type IV Generalized logistic distribution](https://en.wikipedia.org/wiki/Generalized_logistic_distribution#Type_IV).<br>It is defined as the distribution of the random variable `log(Y/(1-Y))`, where `Y ~ Beta(α,β)`.<br>The implementation follows the Distribution.jl interface.

## Installation
Install from the Julia package manager (via Github) by typing `]` in the Julia REPL:
```
] add git@github.com:compbayes/LogisticBetaDistribution.jl.git
```

## Example
```julia
using LogisticBetaDistribution
d = LogisticBeta(1/2, 1/2)
mean(d)    
rand(d, 10)                                 # 10 random draws
pdf(d, 0.5)                                 # pdf at 0.5
cdf(d, 0.5)                                 # cdf at 0.5
d_general = 1 + 2*LogisticBeta(1/2, 1/2)    # Generalized Logistic Beta (1/2,1/2,1,2)
mean(d_general)
```