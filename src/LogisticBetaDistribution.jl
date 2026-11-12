module LogisticBetaDistribution

import Base: rand
using Random
import Random: default_rng, rand!
import Statistics: mean, median, quantile, std, var
import StatsBase: mode, params, params!
using Distributions, SpecialFunctions, LogExpFunctions


include("LogisticBeta.jl")

export mean, median, quantile, std, var, mode, params, rand, rand!, skewness
export kurtosis
export pdf, cdf, ccdf, logpdf, logcdf, gradlogpdf
export entropy, mgf, cgf
export LogisticBeta

end # End module
