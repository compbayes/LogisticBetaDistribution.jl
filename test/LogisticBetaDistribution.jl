using LogisticBetaDistribution
using Distributions: mean, median, std, var, pdf, logpdf, cdf, logcdf, quantile, skewness, partype
using SpecialFunctions: polygamma, trigamma

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
    params(LogisticBeta(2*β, β)) == (2*β, β)
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

end