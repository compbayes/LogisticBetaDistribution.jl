import Distributions: logpdf, pdf, cdf, quantile, std, mean, skewness
import Distributions: kurtosis, partype, @check_args, @distr_support
import Distributions: ccdf, entropy, mgf, cgf, gradlogpdf
import Base.rand

""" 
    LogisticBeta(α, β) 

The Logistic-Beta distribution, also known Generalized Logistic distribution of Type IV.

```math
f(x) = \\frac{1}{\\mathrm{B}(\\alpha,\\beta)} 
       \\frac{e^{-\\beta x}}{(1 + e^{-x})^{(\\alpha+\\beta)}}, \\quad -\\infty < x < \\infty
```
 
The distribution is defined by:

If Y ∼ Beta(α, β), then X = log(Y/(1-Y)) ∼ LogisticBeta(α, β).

The general LogisticBeta(α, β, μ, σ) is obtained by the location-scale construction: 

LogisticBeta(α, β, μ, σ) = μ + σ*LogisticBeta(α, β)




```julia
d = LogisticBeta(1/2, 1/2)   # LogisticBeta distribution with shape parameters α and β

params(d)          # Get the parameters, i.e. (α and β)
pdf(d, 0.5)        # Probability density function at x = 0.5
mean(d)            # Mean 
```

External links

* [Logistic-Beta distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_logistic_distribution#Type_IV)

# Examples
```julia-repl
julia> d = LogisticBeta(1/2, 1/2)
julia> rand(d, 4)'
1×4 adjoint(::Vector{Float64}) with eltype Float64:
 1.00851  0.640297  0.566234  2.16941
julia> pdf(d, 1)
julia> cdf(d, 1)
julia> dist_general = 3 + 2*LogisticBeta(1/2, 1/2)
julia> pdf(dist_general, 1)
```
"""
struct LogisticBeta{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    LogisticBeta{T}(α::T, β::T) where {T} = new{T}(α, β)
end

function LogisticBeta(α::T, β::T; check_args::Bool=true) where {T<:Real}
    @check_args LogisticBeta (α, α > zero(α))
    @check_args LogisticBeta (β, β > zero(β))
    return LogisticBeta{T}(α, β)
end

LogisticBeta(α::Real, β::Real; check_args::Bool=true) = LogisticBeta(promote(α, β)...;
    check_args=check_args)
LogisticBeta(α::Integer, β::Integer; check_args::Bool=true) = LogisticBeta(
    float(α), float(β); check_args=check_args)

@distr_support LogisticBeta -Inf Inf

Base.eltype(::Type{LogisticBeta{T}}) where {T} = T

#### Conversions
Base.convert(::Type{LogisticBeta{T}}, α::S, β::S) where {T<:Real,S<:Real} =
    LogisticBeta(T(α), T(β))
Base.convert(::Type{LogisticBeta{T}}, d::LogisticBeta) where {T<:Real} =
    LogisticBeta{T}(T(d.α), T(d.β))
Base.convert(::Type{LogisticBeta{T}}, d::LogisticBeta{T}) where {T<:Real} = d

#### Parameters
params(d::LogisticBeta) = (d.α, d.β)
partype(::LogisticBeta{T}) where {T} = T

""" 
    rand(d::LogisticBeta[, n::Integer])

Draw `n` random numbers from the logistic-beta distribution `d`. 
"""
function Base.rand(rng::Random.AbstractRNG, d::LogisticBeta)
    # X = logit(Y) with Y = G₁/(G₁+G₂), Gᵢ ~ Gamma, i.e. X = log(G₁) - log(G₂).
    # Working on the log scale avoids Y underflowing to 0 or 1 (X = ±Inf) for small α, β.
    return _randloggamma(rng, d.α) - _randloggamma(rng, d.β)
end

# Draw log(G) with G ~ Gamma(a, 1). For a < 1, use G = G' * U^(1/a), G' ~ Gamma(a+1),
# U ~ Uniform(0,1), so that log(G) stays finite even when G itself would underflow.
function _randloggamma(rng::Random.AbstractRNG, a::Real)
    if a < 1
        return log(rand(rng, Gamma(a + 1))) + log(rand(rng)) / a
    else
        return log(rand(rng, Gamma(a)))
    end
end
#TODO: Implement rand!(rng::Random.AbstractRNG, d::LogisticBeta, x::AbstractArray)

""" 
    logpdf(d::LogisticBeta, x::Real) 

Compute the logpdf of the logistic-beta distribution `d` at `x`. 
"""
logpdf(d::LogisticBeta, x::Real) = -logbeta(d.α, d.β) + d.α * x -
                                   (d.α + d.β) * log1pexp(x)


""" 
    pdf(d::LogisticBeta, x::Real) 

Compute the pdf of the logistic-beta distribution `d` at `x`. 
"""
pdf(d::LogisticBeta, x::Real) = exp(logpdf(d, x))

""" 
    cdf(d::LogisticBeta, x::Real) 

Compute the cdf of the logistic-beta distribution `d` at `x`. 
"""
cdf(d::LogisticBeta, x::Real) = beta_inc(d.α, d.β, logistic(x), logistic(-x))[1]

"""
    ccdf(d::LogisticBeta, x::Real)

Compute the complementary cdf of the logistic-beta distribution `d` at `x`.
"""
ccdf(d::LogisticBeta, x::Real) = beta_inc(d.α, d.β, logistic(x), logistic(-x))[2]


""" 
quantile(d::LogisticBeta, p::Real) 

Compute the `p`-quantile of the logistic-beta distribution `d`. 
"""
quantile(d::LogisticBeta, p::Real) = logit(quantile(Beta(d.α, d.β), p))

""" 
mean(d::LogisticBeta) 

Compute the mean of the logistic-beta distribution `d`. 
"""
mean(d::LogisticBeta) = digamma(d.α) - digamma(d.β)

""" 
mode(d::LogisticBeta) 

Compute the mode of the logistic-beta distribution `d`. 
"""
mode(d::LogisticBeta) = log(d.α / d.β)

""" 
var(d::LogisticBeta) 

Compute the variance of the logistic-beta distribution `d`. 
"""
var(d::LogisticBeta) = trigamma(d.α) + trigamma(d.β)

""" 
std(d::LogisticBeta) 

Compute the standard deviation of the logistic-beta distribution `d`. 
"""
std(d::LogisticBeta) = sqrt(var(d))

""" 
skewness(d::LogisticBeta) 

Compute the skewness of the logistic-beta distribution `d`. 
"""
skewness(d::LogisticBeta) = (polygamma(2, d.α) - polygamma(2, d.β)) / var(d)^(3 / 2)

""" 
kurtosis(d::LogisticBeta) 

Compute the excess kurtosis of the logistic-beta distribution `d`. 
"""
kurtosis(d::LogisticBeta) = (polygamma(3, d.α) + polygamma(3, d.β)) / var(d)^2

"""
entropy(d::LogisticBeta)

Compute the entropy of the logistic-beta distribution `d`. 
"""
entropy(d::LogisticBeta) = logbeta(d.α,d.β) - d.α*digamma(d.α) - d.β*digamma(d.β) + 
        (d.α+d.β)*digamma(d.α+d.β)

""" 
    mgf(d::LogisticBeta, t::Real)

Compute the moment generating function of the logistic-beta distribution `d` at `t`.
The mgf is finite only for `-α < t < β`; `Inf` is returned outside that interval.
"""
mgf(d::LogisticBeta, t::Real) = exp(cgf(d, t))

"""
    cgf(d::LogisticBeta, t::Real)

Compute the cumulant generating function of the logistic-beta distribution `d` at `t`.
The cgf is finite only for `-α < t < β`; `Inf` is returned outside that interval.
"""
function cgf(d::LogisticBeta, t::Real)
    -d.α < t < d.β || return oftype(float(logbeta(d.α, d.β) + t), Inf)
    return logbeta(d.α + t, d.β - t) - logbeta(d.α, d.β)
end

"""
    gradlogpdf(d::LogisticBeta, x::Real)

Compute the gradient of the logpdf of the logistic-beta distribution `d` at `x`.
"""
gradlogpdf(d::LogisticBeta, x::Real) = d.α - (d.α + d.β) * logistic(x)
