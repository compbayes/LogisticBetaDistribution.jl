module LogisticBetaDistribution

import Base: rand
using Random
import Random: default_rng, rand!
import Statistics: mean, median, quantile, std, var
import StatsBase: mode, params, params!
using Distributions, SpecialFunctions, LogExpFunctions


include("LogisticBeta.jl")

export mean, median, quantile, std, var, mode, params, params!, rand, rand!, skewness
export kurtosis
export pdf, cdf, logpdf, logcdf
export LogisticBeta

end # End module
