"""
    JuMPNorm

Abstract type for norms to a JuMP model.
"""
abstract type JuMPNorm end

"""
    L1NormLinear
    L1NormLinear()

Add a set of linear constraints to a model to map an expression to a variable which is the L1 norm of that expression.
"""
struct L1NormLinear <: JuMPNorm end
"""
    MaxNormLinear
    MaxNormLinear()

Add a set of linear constraints to a model to map an expression to a variable which is the max norm of that expression.
"""
struct MaxNormLinear <: JuMPNorm end

"""
    ScaleNorm{S<:Real,N} <: JuMPNorm
    ScaleNorm(scale, n)

Scales result from `n` with a factor `scale`.
"""
struct ScaleNorm{S<:Real,N<:JuMPNorm} <: JuMPNorm
    scale::S
    n::N
end

"""
    SumNorm{N<:JuMPNorm} <: JuMPNorm

Sum of `ns`.
"""
struct SumNorm{N<:JuMPNorm} <: JuMPNorm
    ns::Vector{N}
end
SumNorm(ns::JuMPNorm...) = SumNorm(collect(ns))
SumNorm(sns::Pair{<:Real, <:JuMPNorm}...) = SumNorm(ScaleNorm.(first.(sns), last.(sns))...)



"""
    norm!(s::L1NormLinear, model, X)

Add a set of linear constraints to a model to map `X` to an expression `X′` which is the L1 norm of `X`.

Note that it only works for the objective function and only for minimization.
"""
function norm!(::L1NormLinear, model, X, denom=1)
    # Use trick from http://lpsolve.sourceforge.net/5.1/absolute.htm to make min abs(expression) linear
    X′ = @variable(model, [1:length(X)])
    @constraint(model,  X .<= X′ .* denom)
    @constraint(model, -X .<= X′ .* denom)
    return @expression(model, sum(X′))
end

"""
    norm!(s::L1NormLinear, model, X)

Add a set of linear constraints to a model to map `X` to a variable `X′` which is the max norm of `X`.

Note that it only works for the objective function and only for minimization.
"""
function norm!(::MaxNormLinear, model, X, denom=1)
    # Use trick from https://math.stackexchange.com/questions/2589887/how-can-the-infinity-norm-minimization-problem-be-rewritten-as-a-linear-program to make min abs(expression) linear
    X′ = @variable(model)
    @constraint(model,  X .<= X′ .* denom)
    @constraint(model, -X .<= X′ .* denom)
    return X′
end

function norm!(s::ScaleNorm, model, X, denom=1)
    X′ = norm!(s.n, model, X, denom)
    return @expression(model, s.scale * X′)
end

norm!(s::SumNorm, model, X, denom=1) = mapfoldl(n -> norm!(n, model, X, denom), (X′,X″) -> @expression(model, X′+X″), s.ns, init=@expression(model, 0))
