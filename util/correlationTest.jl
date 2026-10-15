#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using HypothesisTests, StatsBase, Distributions
import HypothesisTests: HypothesisTest, testname, population_param_of_interest, default_tail, show_params, pvalue
using Statistics: cor


# ---------------------------------------------------------------------------
# Spearman's rho (ties handled via mid-ranks), t-approximation with n-2 df
# ---------------------------------------------------------------------------
struct SpearmanCorrelationTest{T<:Real} <: HypothesisTest
    r::T     # Spearman's rho
    n::Int   # number of observations
    t::T     # t-statistic
end

function SpearmanCorrelationTest(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    n = length(x)
    n > 2 || throw(ArgumentError("need at least 3 observations"))
    r = cor(tiedrank(x), tiedrank(y))          # Pearson on mid-ranks = Spearman with ties
    t = r * sqrt((n - 2) / (1 - r^2))
    return SpearmanCorrelationTest(r, n, t)
end

HypothesisTests.testname(::SpearmanCorrelationTest) = "Spearman rank correlation test"
HypothesisTests.population_param_of_interest(x::SpearmanCorrelationTest) = ("Spearman's rho", zero(x.r), x.r)
HypothesisTests.default_tail(::SpearmanCorrelationTest) = :both
HypothesisTests.nobs(x::SpearmanCorrelationTest) = x.n
HypothesisTests.dof(x::SpearmanCorrelationTest) = x.n - 2
HypothesisTests.pvalue(x::SpearmanCorrelationTest; tail=:both) = pvalue(TDist(dof(x)), x.t; tail=tail)

function HypothesisTests.show_params(io::IO, x::SpearmanCorrelationTest, indent="")
    println(io, indent, "number of observations: ", nobs(x))
    println(io, indent, "t-statistic:            ", x.t)
    println(io, indent, "degrees of freedom:     ", dof(x))
end

# ---------------------------------------------------------------------------
# Kendall's tau-b (tie-corrected), normal approximation with tie-corrected
# variance of S = n_concordant - n_discordant
# ---------------------------------------------------------------------------
struct KendallTauTest{T<:Real} <: HypothesisTest
    τ::T     # Kendall's tau-b
    n::Int   # number of observations
    S::T     # n_concordant - n_discordant
    z::T     # z-statistic
end

function KendallTauTest(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    n = length(x)
    n > 2 || throw(ArgumentError("need at least 3 observations"))

    τ = corkendall(x, y)                       # StatsBase returns tau-b

    tx = collect(values(countmap(x)))          # tie-group sizes in x
    ty = collect(values(countmap(y)))          # tie-group sizes in y

    n0 = n * (n - 1) / 2
    n1 = sum(t -> t * (t - 1) / 2, tx)
    n2 = sum(u -> u * (u - 1) / 2, ty)
    S = τ * sqrt((n0 - n1) * (n0 - n2))

    v0 = n * (n - 1) * (2n + 5)
    vt = sum(t -> t * (t - 1) * (2t + 5), tx)
    vu = sum(u -> u * (u - 1) * (2u + 5), ty)
    v1 = sum(t -> t * (t - 1), tx) * sum(u -> u * (u - 1), ty)
    v2 = sum(t -> t * (t - 1) * (t - 2), tx) * sum(u -> u * (u - 1) * (u - 2), ty)

    varS = (v0 - vt - vu) / 18 +
        v1 / (2n * (n - 1)) +
        v2 / (9n * (n - 1) * (n - 2))

    z = S / sqrt(varS)
    return KendallTauTest(τ, n, S, z)
end

HypothesisTests.testname(::KendallTauTest) = "Kendall rank correlation test (tau-b)"
HypothesisTests.population_param_of_interest(x::KendallTauTest) = ("Kendall's tau-b", zero(x.τ), x.τ)
HypothesisTests.default_tail(::KendallTauTest) = :both
HypothesisTests.nobs(x::KendallTauTest) = x.n
HypothesisTests.pvalue(x::KendallTauTest; tail=:both) = pvalue(Normal(), x.z; tail=tail)

function HypothesisTests.show_params(io::IO, x::KendallTauTest, indent="")
    println(io, indent, "number of observations: ", nobs(x))
    println(io, indent, "S (conc. - disc.):      ", x.S)
    println(io, indent, "z-statistic:            ", x.z)
end

# ---------------------------------------------------------------------------
# Wilcoxon Signed-Rank Test
# ---------------------------------------------------------------------------
struct WilcoxonTest{S<:HypothesisTest,T<:Real} <: HypothesisTest
    test::S     # underlying SignedRankTest (exact or approximate)
    n::Int      # number of observations
    p::T        # two-sided p-value
end

function WilcoxonTest(a, b)

    test = SignedRankTest(a, b)
    return WilcoxonTest(test, length(a), pvalue(test))

end

# forward display and p-value to the underlying SignedRankTest
Base.show(io::IO, x::WilcoxonTest) = show(io, x.test)
pvalue(x::WilcoxonTest; tail=:both) = pvalue(x.test; tail=tail)

# ---------------------------------------------------------------------------
# Top-k overlap: fraction of the k best entries w.r.t. x that are also among
# the k best entries w.r.t. y (best = smallest by default, e.g. losses).
# Under h_0 (unrelated rankings) the overlap count is
# Hypergeometric(k, n-k, k) with expectation k^2/n -> exact p-value.
# Ties at the k-th position are broken by index order (stable sortperm).
# ---------------------------------------------------------------------------
struct TopKOverlapTest <: HypothesisTest
    k::Int                  # size of the top set
    n::Int                  # number of observations
    m::Int                  # number of common entries in both top-k sets
    overlap::Float64        # m / k
    common::Vector{Int}     # indices of the common entries
end

function TopKOverlapTest(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, k::Integer; smallest::Bool=true)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    n = length(x)
    1 <= k <= n || throw(ArgumentError("k must be in 1:$n"))

    topx = sortperm(x; rev=!smallest)[1:k]
    topy = sortperm(y; rev=!smallest)[1:k]
    common = sort(intersect(topx, topy))
    m = length(common)

    return TopKOverlapTest(k, n, m, m / k, common)
end

HypothesisTests.testname(x::TopKOverlapTest) = "Top-$(x.k) overlap test (hypergeometric)"
HypothesisTests.population_param_of_interest(x::TopKOverlapTest) = ("Top-$(x.k) overlap fraction", x.k / x.n, x.overlap)
HypothesisTests.default_tail(::TopKOverlapTest) = :right
HypothesisTests.nobs(x::TopKOverlapTest) = x.n

function HypothesisTests.pvalue(x::TopKOverlapTest; tail=:right)
    d = Hypergeometric(x.k, x.n - x.k, x.k)
    if tail == :right
        return ccdf(d, x.m - 1)     # P(M >= m)
    elseif tail == :left
        return cdf(d, x.m)          # P(M <= m)
    elseif tail == :both
        return min(1.0, 2 * min(ccdf(d, x.m - 1), cdf(d, x.m)))
    else
        throw(ArgumentError("tail must be :right, :left or :both, got $tail"))
    end
end

function HypothesisTests.show_params(io::IO, x::TopKOverlapTest, indent="")
    println(io, indent, "number of observations: ", nobs(x))
    println(io, indent, "k:                      ", x.k)
    println(io, indent, "common entries:         ", x.m, " of ", x.k)
    println(io, indent, "expected under h_0:     ", x.k^2 / x.n)
    println(io, indent, "common indices:         ", x.common)
end

