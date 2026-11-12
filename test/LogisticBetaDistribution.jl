using LogisticBetaDistribution
using Distributions: mean, median, std, var, pdf, logpdf, cdf, logcdf, quantile, skewness, partype
using Distributions: ccdf, kurtosis, entropy, mgf, cgf, gradlogpdf, Beta, Logistic
using SpecialFunctions: polygamma, trigamma, digamma
using Random

@testset "LogisticBetaTests.jl" begin

    d = LogisticBeta(1/2,1/2)
    @test cdf.(d, quantile.(d, 0.1:0.1:0.9)) ≈ 0.1:0.1:0.9

    d = LogisticBeta(3/2,3/2)
    @test cdf.(d, quantile.(d, 0.1:0.1:0.9)) ≈ 0.1:0.1:0.9

    @test pdf(d, 1) ≈ exp(logpdf(d, 1))

    @test pdf(d, 1) ≈ pdf(d, -1) # symmetry test

    @test cdf(d,-1) ≈ 1 - cdf(d, 1) # symmetry test

    @test var(d) ≈ std(d).^2

    @test length(rand(d, 4)) == 4

    β = rand()
    @test params(LogisticBeta(2*β, β)) == (2*β, β)
    @test mode(LogisticBeta(2*β, β)) ≈ log(2) 
    @test mean(LogisticBeta(1, 2)) ≈ -1

    @test params(LogisticBeta(0.4,5)) === (0.4,5.0)
    @test LogisticBeta(1, 2) == LogisticBeta(1.0, 2.0)
    @test LogisticBeta(1, 2.0) == LogisticBeta(1.0, 2.0)

    # Z-distribution, location scale variant
    x = 1; μ = 2; σ = 3;
    @test mean(μ + LogisticBeta(1/2,1/2)) ≈ μ 
    @test std(LogisticBeta(1/2, 1/2)*σ) ≈ σ*std(LogisticBeta(1/2,1/2))
    
    @test pdf(μ + LogisticBeta(1/2,1/2)*σ, x) ≈ (1/σ)*pdf(LogisticBeta(1/2,1/2), (x-μ)/σ)
    @test cdf(μ + LogisticBeta(1/2,1/2)*σ, x) ≈ cdf(LogisticBeta(1/2,1/2), (x-μ)/σ)
    @test skewness(μ + LogisticBeta(1/2,1/2)*σ) ≈ skewness(LogisticBeta(1/2,1/2))

    # Parameter validation
    @test_throws DomainError LogisticBeta(0.0, 1.0)
    @test_throws DomainError LogisticBeta(1.0, 0.0)
    @test_throws DomainError LogisticBeta(-1.0, 1.0)
    @test_throws DomainError LogisticBeta(1.0, -1.0)

    # eltype
    @test Base.eltype(LogisticBeta{Float64}) === Float64
    @test Base.eltype(LogisticBeta{Float32}) === Float32

    # partype
    @test LogisticBetaDistribution.partype(LogisticBeta(1.0, 2.0)) === Float64

    # convert
    d64 = LogisticBeta(1.0, 2.0)
    d32 = convert(LogisticBeta{Float32}, d64)
    @test d32.α === Float32(1.0)
    @test d32.β === Float32(2.0)
    @test convert(LogisticBeta{Float64}, d64) === d64

    # convert from raw parameters (line 72)
    d_conv = LogisticBetaDistribution.convert(LogisticBeta{Float32}, 1.0, 2.0)
    @test d_conv.α === Float32(1.0)
    @test d_conv.β === Float32(2.0)

    # median
    @test isapprox(median(LogisticBeta(3/2, 3/2)), 0.0, atol=1e-14)

    # logcdf
    d_asym = LogisticBeta(2.0, 3.0)
    @test logcdf(d_asym, 0.5) ≈ log(cdf(d_asym, 0.5))

    # skewness with asymmetric parameters
    α_s, β_s = 2.0, 5.0
    d_skew = LogisticBeta(α_s, β_s)
    expected_skewness = (polygamma(2, α_s) - polygamma(2, β_s)) / (trigamma(α_s) + trigamma(β_s))^(3/2)
    @test skewness(d_skew) ≈ expected_skewness


    # LogisticBeta(1, 1) is the standard logistic distribution
    d1 = LogisticBeta(1, 1)
    dl = Logistic(0.0, 1.0)
    for x in (-3.0, -0.5, 0.0, 1.2, 4.0)
        @test pdf(d1, x) ≈ pdf(dl, x)
        @test cdf(d1, x) ≈ cdf(dl, x)
        @test ccdf(d1, x) ≈ ccdf(dl, x)
    end
    @test quantile(d1, 0.3) ≈ quantile(dl, 0.3)
    @test var(d1) ≈ π^2 / 3
    @test isapprox(skewness(d1), 0.0, atol=1e-14)
    @test kurtosis(d1) ≈ 1.2
    @test entropy(d1) ≈ 2.0
    @test mgf(d1, 0.4) ≈ π * 0.4 / sin(π * 0.4)

    # pdf/logpdf are numerically stable far out in the tails
    @test isfinite(logpdf(d_asym, 1000.0)) && isfinite(logpdf(d_asym, -1000.0))
    @test pdf(d_asym, 1000.0) == 0.0
    @test !isnan(pdf(LogisticBeta(200.0, 300.0), 0.0))

    # ccdf
    for x in (-2.0, 0.0, 0.7, 3.0)
        @test ccdf(d_asym, x) ≈ 1 - cdf(d_asym, x)
        @test ccdf(d_asym, x) ≈ cdf(LogisticBeta(3.0, 2.0), -x) # X ~ LB(α,β) ⇒ -X ~ LB(β,α)
    end
    @test ccdf(d_asym, 40.0) > 0         # where 1 - cdf underflows to 0
    @test 1 - cdf(d_asym, 40.0) == 0

    # kurtosis matches the cumulant formula κ₄/κ₂²
    @test kurtosis(d_skew) ≈ (polygamma(3, α_s) + polygamma(3, β_s)) /
                             (trigamma(α_s) + trigamma(β_s))^2

    # entropy: change of variables from Y ~ Beta(α, β), X = logit(Y)
    # H(X) = H(Y) - E[log Y] - E[log(1-Y)]
    for (a, b) in ((0.5, 0.5), (2.0, 3.0), (7.0, 1.5))
        ψab = digamma(a + b)
        @test entropy(LogisticBeta(a, b)) ≈
              entropy(Beta(a, b)) - (digamma(a) - ψab) - (digamma(b) - ψab)
    end

    # entropy vs Monte Carlo estimate -E[logpdf(X)]
    rng = MersenneTwister(123)
    xs = rand(rng, d_skew, 200_000)
    @test isapprox(-sum(logpdf.(d_skew, xs)) / length(xs), entropy(d_skew), rtol=1e-2)

    # rand: sample moments match theoretical moments
    @test isapprox(sum(xs) / length(xs), mean(d_skew), atol=0.01)
    @test isapprox(sum(abs2, xs .- mean(d_skew)) / length(xs), var(d_skew), rtol=0.02)
    @test rand(rng, d_skew) isa Float64

    # rand stays finite and correct for small shape parameters
    d_small = LogisticBeta(0.05, 0.2)
    xs_small = rand(rng, d_small, 200_000)
    @test all(isfinite, xs_small)
    @test all(isfinite, rand(rng, LogisticBeta(0.01, 0.01), 10_000))
    @test isapprox(sum(xs_small) / length(xs_small), mean(d_small), rtol=0.02)
    @test isapprox(sum(xs_small .<= 0.0) / length(xs_small), cdf(d_small, 0.0), atol=0.005)

    # mgf / cgf
    @test mgf(d_asym, 0.0) ≈ 1
    @test cgf(d_asym, 0.0) ≈ 0 atol=1e-14
    @test cgf(d_asym, 0.8) ≈ log(mgf(d_asym, 0.8))
    @test mgf(d_asym, 1.0) ≈ 1           # B(3,2)/B(2,3) = 1
    h = 1e-4 # cumulants from finite differences of the cgf
    @test (cgf(d_asym, h) - cgf(d_asym, -h)) / (2h) ≈ mean(d_asym) rtol=1e-6
    @test (cgf(d_asym, h) - 2cgf(d_asym, 0.0) + cgf(d_asym, -h)) / h^2 ≈ var(d_asym) rtol=1e-4
    @test isapprox(mgf(d_skew, 0.5), sum(exp.(0.5 .* xs)) / length(xs), rtol=1e-2)
    # mgf only exists for -α < t < β
    @test mgf(d_asym, 3.0) == Inf
    @test mgf(d_asym, -2.0) == Inf
    @test cgf(d_asym, 5.0) == Inf
    @test isfinite(mgf(d_asym, 2.99)) && isfinite(mgf(d_asym, -1.99))

    # gradlogpdf vs finite difference of logpdf, and zero at the mode
    for x in (-3.0, -0.2, 0.0, 1.5, 6.0)
        @test gradlogpdf(d_asym, x) ≈ (logpdf(d_asym, x + h) - logpdf(d_asym, x - h)) / (2h) rtol=1e-6
    end
    @test isapprox(gradlogpdf(d_asym, mode(d_asym)), 0.0, atol=1e-12)

end
