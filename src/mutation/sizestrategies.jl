

"""
    AbstractΔSizeStrategy

Abstract base type for strategies for how to change the size.

See [`Δsize!`](@ref).
"""
abstract type AbstractΔSizeStrategy end

"""
    AbstractJuMPΔSizeStrategy <: AbstractΔSizeStrategy

Abstract type for strategies to change or align the sizes of vertices using JuMP.
"""
abstract type AbstractJuMPΔSizeStrategy <: AbstractΔSizeStrategy end

"""
    DecoratingJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy

Abstract type for [`AbstractJuMPΔSizeStrategies`](@ref AbstractJuMPΔSizeStrategy) which wants to delegate some parts of the problem formulation to another strategy.

More concretely: If `s` is a [`DecoratingJuMPΔSizeStrategy`](@ref) then [`base(s)`](@ref base(s::DecoratingJuMPΔSizeStrategy)) will be used unless explicitly stated through dispatch.
"""
abstract type DecoratingJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy end

"""
    base(s::DecoratingJuMPΔSizeStrategy)

Return the strategy wrapped by `s`.
"""
base(s::DecoratingJuMPΔSizeStrategy) = s.base # Just make a wild guess rather than throwing an error

"""
    ΔSizeFailError <: Exception

Size change could not be solved.
"""
struct ΔSizeFailError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ΔSizeFailError) = print(io, "ΔSizeFailError: ", e.msg)

"""
    ThrowΔSizeFailError <: AbstractJuMPΔSizeStrategy
    ThrowΔSizeFailError(msg::String)

Throws an `ΔSizeFailError` with message `msg`.
"""
struct ThrowΔSizeFailError{F} <: AbstractJuMPΔSizeStrategy
    msgfun::F
end
ThrowΔSizeFailError(msg::AbstractString) = ThrowΔSizeFailError(vs -> msg)
ThrowΔSizeFailError() = ThrowΔSizeFailError(vs -> string("Could not change size of vertices ", join(nameorrepr.(vs), ", ", " and "),"!")) 

add_participants!(::ThrowΔSizeFailError, vs=AbstractVertex[]) = vs

"""
    ΔSizeFailNoOp <: AbstractJuMPΔSizeStrategy
    ΔSizeFailNoOp()

Does not perform any action.
"""
struct ΔSizeFailNoOp <: AbstractJuMPΔSizeStrategy end

add_participants!(::ΔSizeFailNoOp, vs=AbstractVertex[]) = vs

"""
    LogΔSizeExec <: AbstractJuMPΔSizeStrategy
    LogΔSizeExec(msgfun)
    LogΔSizeExec(msgfun, level::Logging.LogLevel)
    LogΔSizeExec(msgfun, level::Logging.LogLevel, andthen::AbstractJuMPΔSizeStrategy)

Logs `msgfun(v)` at log level `level`, then executes [`AbstractJuMPΔSizeStrategy`](@ref) `andthen` for vertex `v`.
"""
struct LogΔSizeExec{F, S} <: DecoratingJuMPΔSizeStrategy
    msgfun::F
    level::LogLevel
    andthen::S
end
LogΔSizeExec(msgfun, level=Logging.Info) = LogΔSizeExec(msgfun, level, ΔSizeFailNoOp())
LogΔSizeExec(msg::String, level::LogLevel=Logging.Info, andthen=ΔSizeFailNoOp()) = LogΔSizeExec(v -> msg, level, andthen)
LogSelectionFallback(nextstr, andthen; level=Logging.Warn) = LogΔSizeExec(v -> "Size change for vertex $(nameorrepr(v)) failed! $nextstr", level, andthen)
base(s::LogΔSizeExec) = s.andthen
fallback(s::LogΔSizeExec) = fallback(base(s))

add_participants!(s::LogΔSizeExec, vs=AbstractVertex[]) = add_participants!(s.andthen, vs)


"""
    DefaultJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy
    DefaultJuMPΔSizeStrategy()

Default strategy which applies default constraints or objectives to a model. 

Intended use is for strategies which apply special constraints or objectives to a subset of the vertices.

Also useful when configuring [`DecoratingJuMPΔSizeStrategies`](@ref DecoratingJuMPΔSizeStrategy).

### Examples

Use to get add special constraints for a subset of vertices:
```julia
function NaiveNASlib.addsomeconstraint!(s::MySpecialStrategy, v::AbstractVertex, data)
    if v === s.targetvertex
        # Add some special constraints here
    else
        # Use the default
        NaiveNASlib.addsomeconstraint!(DefaultJuMPΔSizeStrategy(), v::AbstractVertex, data)
    end
end
```
Use with a [`DecoratingJuMPΔSizeStrategy`](@ref):
```jldoctest
julia> using NaiveNASlib, NaiveNASlib.Advanced

julia> WithUtilityFun(defaultutility, DefaultJuMPΔSizeStrategy())
WithUtilityFun{typeof(defaultutility), DefaultJuMPΔSizeStrategy}(NaiveNASlib.defaultutility, DefaultJuMPΔSizeStrategy())
```
"""
struct DefaultJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy end
fallback(::DefaultJuMPΔSizeStrategy) = default_sizefallback("Solver failed!")

"""
    TimeLimitΔSizeStrategy{S} <: DecoratingJuMPΔSizeStrategy
    TimeLimitΔSizeStrategy(limit::Number)
    TimeLimitΔSizeStrategy(limit::Number, base::S)

Sets the time limit for the solver to `limit`. Use strategy `base` for all other aspects. 
"""
struct TimeLimitΔSizeStrategy{S,F} <: DecoratingJuMPΔSizeStrategy
    limit::Float64
    base::S
    fallback::F
end
TimeLimitΔSizeStrategy(limit::Number, base=DefaultJuMPΔSizeStrategy(); fallback=default_sizefallback("Solver failed!")) = TimeLimitΔSizeStrategy(limit, base, fallback)
TimeLimitΔSizeStrategy(limit::Number, base::S, fallback::F) where {S,F} = TimeLimitΔSizeStrategy{S,F}(Float64(limit), base, fallback)
base(s::TimeLimitΔSizeStrategy) = s.base
fallback(s::TimeLimitΔSizeStrategy) = s.fallback
add_participants!(s::TimeLimitΔSizeStrategy, vs=AbstractVertex[]) = add_participants!(base(s), vs)

"""
    TimeOutAction{S,A,F} <: DecoratingJuMPΔSizeStrategy
    TimeOutAction(action::A, base::S, fallback::F)
    TimeOutAction(base; fallback)

Calls `action(model)` if JuMP model `model` has status `MOI.TIME_LIMIT` after optimization stops. Use strategy `base` for all other aspects.

Default action is to display a warning and then apply `fallback` (default [`ΔSizeFailNoOp`](@ref)).
"""
struct TimeOutAction{S,A,F} <: DecoratingJuMPΔSizeStrategy
    action::A
    base::S
    fallback::F
end
base(s::TimeOutAction) = s.base
function TimeOutAction(;action=default_timeout, base::AbstractJuMPΔSizeStrategy=DefaultJuMPΔSizeStrategy(), fallback=default_sizefallback("Solver failed!")) 
    TimeOutAction(action, base, fallback)
end
fallback(s::TimeOutAction) = s.fallback
function default_timeout(args...) 
    @warn "Solver timed out"
    return false
end
add_participants!(s::TimeOutAction, vs=AbstractVertex[]) = add_participants!(base(s), vs)



struct Exact end
struct Relaxed end

"""
    ΔNout <: AbstractJuMPΔSizeStrategy
    ΔNout{T, V, F}(Δs::Dict{V, Int}, fallback::F)

Strategy for changing nout of vertices.
    
For each key-value pair in `v, Δ` in `Δs`, change [`nout(v)`](@ref) by `Δ`, i.e new size is `nout(v) + Δ`.

If `T == Exact`, size change will be added as a constraint to the model which means that the operation will fail
if it is not possible to change `nout(v)` by exactly `Δ`.

If `T == Relaxed`, size change will be added as an objective to the model which means that `nout(v)` might 
not change by exactly `Δ`. In addition, a constraint that `nout(v)` must change is also added.
"""
struct ΔNout{T, V, F} <: AbstractJuMPΔSizeStrategy
    Δs::Dict{V, Int}
    fallback::F
end
fallback(s::ΔNout) = s.fallback

add_participants!(s::ΔNout, vs=AbstractVertex[]) = append!(vs, filter(v -> v ∉ vs, all_in_Δsize_graph(keys(s.Δs), Output())))


generic_Δnin_docstring_examples(fname::String; kwargs...) = generic_Δnin_docstring_examples(args ->  "$fname($args)"; kwargs...)
generic_Δnin_docstring_examples(fgen; footer = "```", kwargs...) = generic_Δnout_docstring_examples(fgen; footer="", kwargs...) * """

julia> v3 = conc(affine(iv, 4), affine(iv, 5), affine(iv, 6); dims=2);

julia> v4 = conc(affine(iv, 7), affine(iv, 8); dims=2);

julia> $(fgen("v3, 3, missing, 2"));

julia> $(fgen("v3 => (3, missing, 2), v4 => (-2, 0)"));
""" * footer

generic_Δnout_docstring_examples(fname::String; kwargs...) = generic_Δnout_docstring_examples(args -> "$fname($args)"; kwargs...)
generic_Δnout_docstring_examples(fgen; header="```julia-repl", footer = "```") = """
### Examples
$header
julia> using NaiveNASlib, NaiveNASlib.Extend, NaiveNASlib.Advanced

julia> module TestNNLib
           using NaiveNASlib, NaiveNASlib.Extend
           mutable struct Affine{T} W::Matrix{T} end
           NaiveNASlib.nout(a::Affine) = size(a.W, 1)
           NaiveNASlib.nin(a::Affine) = [size(a.W, 2)]
           NaiveNASlib.Δsize!(a::Affine, ins::AbstractVector, outs::AbstractVector) = a.W = parselect(a.W, 2=>ins[1], 1=>outs)
           affine(in, outsize) = absorbvertex(Affine(ones(outsize, nout(in))), in)
           export affine
       end;

julia> using .TestNNLib;

julia> iv = affine(inputvertex("in", 3), 6);

julia> v1 = affine(iv, 4);

julia> v2 = affine(iv, 5);

julia> $(fgen("v1, 3"));

julia> $(fgen("v1 => -3, v2 => -2"));

julia> $(fgen("Dict(v1 => 3, v2 => 2)"));
""" * footer

"""
    ΔNout{T}(args...;fallback::F)

Return a `ΔNout{T, V, F}` created from `args` where `args` can be either a `Dict` or arguments used to create a `Dict`
(e.g. a set of `Pair`s or a generator).

Also helps creating a `Dict{V, Int}` from e.g. a `Dict{Any, Any}` as a convenience.

`T` is typically either `Exact` or `Relaxed`.

$(generic_Δnout_docstring_examples(args -> "ΔNout{NaiveNASlib.Exact}($args; fallback=ΔSizeFailNoOp())"; header="```jldoctest"))
"""
ΔNout{T}(v::AbstractVertex, Δ::Integer; fallback) where T = ΔNout{T}(v=>Int(Δ); fallback)
ΔNout{T}(args...; fallback) where T = ΔNout{T}(Dict(args...); fallback)
ΔNout{T}(Δs::AbstractDict; fallback) where T = ΔNout{T}(Dict(k=>Int(v) for (k,v) in Δs); fallback)
ΔNout{T}(Δs::AbstractDict{V, Int}; fallback::F) where {T,V,F} = ΔNout{T,V,F}(Dict(Δs), fallback)

"""
    ΔNoutExact(args...; fallback)

Return a `ΔNout{Exact}` with `fallback` set to give a warning and then try again with [`ΔNoutRelaxed(args...)`](@ref).

Accepts either a `Dict` or arguments used to create a `Dict` (e.g. a set of `Pair`s or a generator).

$(generic_Δnout_docstring_examples("ΔNoutExact"; header="```jldoctest"))
"""
ΔNoutExact(args...; fallback=default_noutfallback(ΔNoutRelaxed,"nout", args)) = ΔNout{Exact}(args...;fallback)

"""
    ΔNoutRelaxed(args...;fallback)

Return a `ΔNout{Relaxed}` with `fallback` set to [`ΔSizeFailNoOp`](@ref).

Accepts either a `Dict` or arguments used to create a `Dict` (e.g. a set of `Pair`s or a generator).

$(generic_Δnout_docstring_examples("ΔNoutRelaxed"; header="```jldoctest"))
"""
ΔNoutRelaxed(args...;fallback=default_noutfallback("nout", args)) = ΔNout{Relaxed}(args...;fallback)

const Maybe{T} = Union{T, Missing}

Δnout_err_info(v, Δ::Union{<:Maybe{Int}, <:Pair}...) = "$(nameorrepr(v)) by $(join(first.(Δ), ", ", " and "))"
Δnout_err_info(v, Δ::Tuple) = Δnout_err_info(v, Δ...)
Δnout_err_info(ps::Pair...) = join(map(p ->Δnout_err_info(p...), ps), ", ", " and ")
Δnout_err_info(d::AbstractDict) = join((Δnout_err_info(k, v) for (k,v) in d), ", ", " and ")

function default_noutfallback(nextfallback, dirstr, args) 
    msgfun = v -> string("Could not change ", dirstr, " of ", Δnout_err_info(args...), "! Relaxing constraints...")
    return LogΔSizeExec(msgfun, Logging.Warn, nextfallback(args...))
end

default_noutfallback(dirstr, args) = LogΔSizeExec(vs -> string("Could not change ", dirstr, " of ", Δnout_err_info(args...), "! Attempt aborted!"), Logging.Warn)
default_sizefallback(msg) = LogΔSizeExec(msg, Logging.Warn)

"""
    ΔNinExact(args...; fallback)

Return a `ΔNout{Exact}` configured to set `nout` of the inputs to vertices in `args` to the given sizes.

For vertices with more than one input, the size change must be expressed as a tuple with one element per input. 
Use `missing` to indicate that no special treatment is needed for an input.

Accepts either a `Dict` or arguments used to create a `Dict` (e.g. a set of `Pair`s or a generator).

By default, `fallback` is set to give a warning and then try again with [`ΔNinRelaxed(args...)`](@ref).

$(generic_Δnin_docstring_examples("ΔNinExact"; header="```jldoctest"))
"""
ΔNinExact(args...; fallback=default_noutfallback(ΔNinRelaxed, "nin", args)) = ΔNout{Exact}(Δnin2Δnout(args...); fallback)

"""
    ΔNinRelaxed(args...; fallback)

Same as [`ΔNinExact`](@ref) except `Exact` is replaced by `Relaxed` and `fallback` set to [`ΔSizeFailNoOp`](@ref) by default.

$(generic_Δnin_docstring_examples("ΔNinRelaxed"; header="```jldoctest"))
"""
ΔNinRelaxed(args...; fallback=default_noutfallback("nout", args)) = ΔNout{Relaxed}(Δnin2Δnout(args...); fallback)

Δnin2Δnout(d::AbstractDict) = reduce(merge!, (Δnin2Δnout(k,v) for (k,v) in d); init=Dict())
Δnin2Δnout(ps::Pair{<:AbstractVertex}...) = mapreduce(p -> Δnin2Δnout(p...), merge!, ps; init=Dict()) 
Δnin2Δnout(v::AbstractVertex, Δs...) = Δnin2Δnout(v, Δs)
Δnin2Δnout(v::AbstractVertex, Δs::Pair{<:Tuple, <:Union{Relaxed, Exact}}) = Dict(k => (v => last(Δs)) for (k,v) in Δnin2Δnout(v, first(Δs))) 
function Δnin2Δnout(v::AbstractVertex, Δs::Tuple)
    @assert length(Δs) == length(inputs(v)) "Must supply same number of Δs as v has inputs! Got $Δs for $(nameorrepr(v)) with $(length(inputs(v))) inputs. 
    $(length(Δs) < length(inputs(v)) ? "Tip: Use missing to indicate that no special requirement shall be used for an input" : "")"
    
    inds = findall(!ismissing, Δs)
    return Dict(inputs(v)[i] => Δs[i] for i in inds)
end

"""
    ΔNoutMix{VE, VR, F} <: AbstractJuMPΔSizeStrategy
    ΔNoutMix(exact::ΔNout{Exact, VE, F}, relax::ΔNout{Relaxed, VR, F}, fallback::F)

Strategy for changing nout of vertices.

Applies both `exact` and `relax`, thereby allowing for having both hard and relaxed size constraints for different vertices.

Can be conveniently created by [`ΔNout`](@ref) or [`ΔNin`](@ref). 
"""
struct ΔNoutMix{VE, VR, F} <: AbstractJuMPΔSizeStrategy
    exact::ΔNout{Exact, VE, F}
    relax::ΔNout{Relaxed, VR, F}
    fallback::F
end
fallback(s::ΔNoutMix) = s.fallback

function add_participants!(s::ΔNoutMix, vs=AbstractVertex[]) 
    add_participants!(s.exact, vs)
    add_participants!(s.relax, vs)
    return vs
end


"""
    relaxed(Δ)

Return `Δ => Relaxed()` which indicates that `Δ` shall be relaxed when changing size.

See [`Δnout!`](@ref) and [`Δnin!`](@ref).
"""
relaxed(Δ::Integer) = Δ => Relaxed()
relaxed(Δs::Tuple{Vararg{Maybe{Int}}}) = Δs => Relaxed()
relaxed(Δs...) = Δs => Relaxed()

"""
    ΔNout(args...;[fallback])

Splits `args` into relaxed and exact size changes and creates the appropriate strategy (one of `ΔNout{Exact}, ΔNout{Relaxed} or ΔNoutMix`).

Fallback strategy may be supplied through the `fallback` keyword. Defaults to `ΔNoutRelaxed` for all given vertices.

$(generic_Δnout_docstring_examples("ΔNout"; header="```jldoctest", footer=""))
julia> ΔNout(v1, relaxed(2));

julia> ΔNout(v1 => relaxed(2), v2 => -1);
``` 
"""
function ΔNout(args...;kwargs...) 
    exact, relaxed = split_exact_relaxed(args)
    return ΔNout(exact, relaxed; kwargs...)
end

"""
    ΔNin(args...;[fallback])

Splits `args` into relaxed and exact size changes and creates the appropriate strategy (one of `ΔNout{Exact}, ΔNout{Relaxed} or ΔNoutMix`).

Fallback strategy may be supplied through the `fallback` keyword. Defaults to `ΔNinRelaxed` for all given vertices.

$(generic_Δnin_docstring_examples("ΔNin"; header="```jldoctest", footer=""))
julia> ΔNin(v3, relaxed(3), missing, 2);

julia> ΔNin(v3 => (relaxed(3), missing, 2), v4 => relaxed(-2, 0));
``` 
"""
function ΔNin(args...;fallback=nothing) 
    exact,relaxed = split_exact_relaxed(Δnin2Δnout(args...)) # To support mixed cases, e.g. Δnin!(v => (2, relaxed(4)))
    fallback = fallback === nothing ? default_noutfallback((args...) -> ΔNoutRelaxed(merge(exact, relaxed); fallback=default_noutfallback("nin", args)), "nin", args) : fallback
    return ΔNout(exact, relaxed; fallback)
end

function ΔNout(exact::Dict, relaxed::Dict; fallback = default_noutfallback(ΔNoutRelaxed, "nout", merge(exact,relaxed)))
    isempty(relaxed) && return ΔNoutExact(exact;fallback)
    isempty(exact) && return ΔNoutRelaxed(relaxed; fallback=_relaxed_fallback(relaxed, fallback))
    return ΔNoutMix(ΔNoutExact(exact;fallback), ΔNoutRelaxed(relaxed;fallback), fallback)
end

_relaxed_fallback(Δs, fb) = _is_same_fallback(Δs, fb) ? fallback(fb) : fb

_is_same_fallback(Δs, fb::LogΔSizeExec) = _is_same_fallback(Δs, base(fb))
_is_same_fallback(Δs, fb::ΔNout{Relaxed}) = fb.Δs == Δs
_is_same_fallback(Δs, fb) = false

split_exact_relaxed(p::Tuple{AbstractVertex, Any}) = split_exact_relaxed(tuple(Pair(p...)))
split_exact_relaxed(d::Tuple{Dict}) = split_exact_relaxed(first(d))
function split_exact_relaxed(args::Union{AbstractDict, Tuple{Vararg{Pair}}})
    exact=Dict()
    relaxed=Dict() 
    for (k, v) in args
        if !isa(v, Pair) || v isa Pair{<:Any, Exact}
            exact[k] = first(v) # first works for numbers too
        elseif v isa Pair{<:Any, Relaxed}
            relaxed[k] = first(v)
        else
            throw(ArgumentError(string("Unhandled value type: ", v)))
        end
    end
    return exact, relaxed
end

"""
    AlignNinToNout{S, F} <: DecoratingJuMPΔSizeStrategy
    AlignNinToNout(;vstrat::S, fallback::F)
    AlignNinToNout(vstrat::S, fallback::F) 

Adds variables and constraints for `nin(vi) == nout.(inputs(vi))`.

Generally useful when doing structural changes such are removing/adding vertices or edges.

If it fails, the operation will be retried with the `fallback` strategy (default [`ΔSizeFailNoOp`](@ref)).
"""
struct AlignNinToNout{S, F} <: DecoratingJuMPΔSizeStrategy
    nindict::Dict{AbstractVertex, Vector{JuMP.VariableRef}}
    vstrat::S
    fallback::F
end
AlignNinToNout(;vstrat=DefaultJuMPΔSizeStrategy(), fallback=default_sizefallback("Failed to align Nin to Nout!!")) = AlignNinToNout(vstrat, fallback)
AlignNinToNout(vstrat, fallback) = AlignNinToNout(Dict{AbstractVertex, Vector{JuMP.VariableRef}}(), vstrat, fallback)
fallback(s::AlignNinToNout) = s.fallback
base(s::AlignNinToNout) = s.vstrat

function add_participants!(s::AlignNinToNout, vs=AbstractVertex[]) 
    add_participants!(base(s), vs)
    append!(vs, filter(v -> v ∉ vs, all_in_Δsize_graph(keys(s.nindict), Both())))
end


"""
    AlignNinToNoutVertices{V1,V2,S,F} <: AbstractJuMPΔSizeStrategy
    AlignNinToNoutVertices(vin::V1, vout::V2, inds; vstrat::S=AlignNinToNout(), fallback::F=ΔSizeFailNoOp())
    AlignNinToNoutVertices(vin::V1, vout::V2, inds, vstrat::S, fallback::F)

Same as [`AlignNinToNout`](@ref) with an additional constraint that [`nin(s.vin)`](@ref)`[s.ininds] ==` [`nout(s.vout)`](@ref) 
where `s` is a [`AlignNinToNoutVertices`](@ref).

Useful in the context of adding edges to vertices to align sizes before the edge has been added.

If it fails, the operation will be retried with the `fallback` strategy (default [`ΔSizeFailNoOp`](@ref)).
"""
struct AlignNinToNoutVertices{V1,V2,S,F} <: DecoratingJuMPΔSizeStrategy
    vin::V1
    vout::V2
    ininds::Vector{Int}
    vstrat::S
    fallback::F
end
AlignNinToNoutVertices(vin, vout, inds; vstrat=AlignNinToNout(), fallback=failtoalign(vin, vout)) = AlignNinToNoutVertices(vin, vout, inds, vstrat, fallback)
AlignNinToNoutVertices(vin, vout, inds, vstrat, fallback) = AlignNinToNoutVertices(vin, vout, collect(Int, inds), vstrat, fallback)

fallback(s::AlignNinToNoutVertices) = s.fallback
base(s::AlignNinToNoutVertices) = s.vstrat

function add_participants!(s::AlignNinToNoutVertices, vs=AbstractVertex[]) 
    add_participants!(base(s), vs)
    append!(v -> v ∉ vs, vcat(all_in_Δsize_graph(s.vin, Input()), all_in_Δsize_graph(s.vout, Output())))
end


failtoalign(vin, vout) = LogΔSizeExec(vs -> string("Could not align nout of ", nameorrepr(vin), " to nin of ", nameorrepr(vout), "!!"), Logging.Warn)

"""
    TruncateInIndsToValid{S} <: AbstractΔSizeStrategy
    TruncateInIndsToValid()
    TruncateInIndsToValid(s::S)

Ensures that all selected input indices are within range of existing input indices after applying `s` 
(default [`DefaultJuMPΔSizeStrategy`](@ref)).

Not needed in normal cases, but certain structural mutations (e.g create_edge!) may cause this to happen 
due to how constraints are (not) created when original sizes do not align in conjunction with how result of 
selection is interpreted.

An example of when it is needed is when adding an edge from vertex `vi` to an invariant vertex `vo` where 
[`nout(vi)`](@ref) `>` [`nout(vo)`](@ref). In this case it is expected that the result of the optimization
is that the indices `1:nout(vi)` of `vi` shall be kept. However, this will propagate to `vo` which will be 
instructed to keep indices it does not have. With this strategy, all indices which are larger than `nout(vo)`
will be replaced by `-1` (which indicates that new parameters shall be created) 

While this may be considered a flaw in the output selection procedure, it is rare enough so that in most cases 
when it happens it is the result of a user error or lower level bug. Therefore this strategy is left optional 
to be used only in cases when mismatches are expected.

Note that in most normal cases, this has no effect since vertices capable of getting new input edges generally 
don't have parameters.
"""
struct TruncateInIndsToValid{S} <: AbstractΔSizeStrategy
    strategy::S
end
TruncateInIndsToValid() = TruncateInIndsToValid(DefaultJuMPΔSizeStrategy())
base(s::TruncateInIndsToValid) = s.strategy

add_participants!(s::TruncateInIndsToValid, vs=AbstractVertex[]) = add_participants!(base(s), vs)


"""
    WithUtilityFun{F, S} <: AbstractΔSizeStrategy
    WithUtilityFun(utilityfun::F, strategy::S)

Applies neuron indices selection with `strategy` and using `utilityfun` to compute the utility of neurons indices.

Note that `utilityfun` will override any utility function supplied in function call. Thus it is possible use 
`WithUtilityFun` to change utility function e.g. when switching to a fallback strategy.
"""
struct WithUtilityFun{F, S} <: AbstractΔSizeStrategy
    utilityfun::F
    strategy::S
end
WithUtilityFun(f) = s -> WithUtilityFun(f ,s) 
base(s::WithUtilityFun) = s.strategy
fallback(s::WithUtilityFun) = fallback(s.strategy)

add_participants!(s::WithUtilityFun, vs=AbstractVertex[]) = add_participants!(base(s), vs)

"""
    UpperInsertBound{V<:AbstractDict{<:AbstractVertex, <:Integer}, S, F} <: DecoratingJuMPΔSizeStrategy
    UpperInsertBound(vbounds; strategy, fallback)
    UpperInsertBound(vs, bound::Integer; strategy, fallback)

Sets upper bound for how many new neurons to insert of each vertex `v` in `vbounds` to `vbounds[v]` and then applies neuron indices 
selection with `strategy` (default [`DefaultJuMPΔSizeStrategy`](@ref)).

Alternatively, sets the upper bound of each vertex in `vs` to `bound`. Applies `bound` to all vertices if `vs` is empty.

Mostly useful to minimize solver time in cases when one does not want any new neurons as disallowing this allows skipping a couple of constraints.

If it fails, the operation will be retried with the `fallback` strategy (default [`ΔSizeFailNoOp`](@ref)).
"""
struct UpperInsertBound{VB, S, F} <: DecoratingJuMPΔSizeStrategy
    vbounds::VB
    strategy::S
    fallback::F
end

function UpperInsertBound(vs::AbstractVector{<:AbstractVertex}, bound::Integer=0; kwargs...)  
    isempty(vs) ? UpperInsertBound(bound; kwargs...) : UpperInsertBound(Dict(v => bound for v in vs); kwargs...)
end
function UpperInsertBound(vbounds::Union{Integer, AbstractDict}; strategy=DefaultJuMPΔSizeStrategy(), fallback=default_sizefallback("Solver failed!")) 
    UpperInsertBound(vbounds, strategy, fallback)
end
base(s::UpperInsertBound) = s.strategy
fallback(s::UpperInsertBound) = s.fallback
add_participants!(s::UpperInsertBound, vs=AbstractVertex[]) = add_participants!(base(s), vs) 

"""
    AbstractAfterΔSizeStrategy <: DecoratingJuMPΔSizeStrategy

Abstract base type for strategies which perform actions after size has changed (e.g validation and logging).
"""
abstract type AbstractAfterΔSizeStrategy <: DecoratingJuMPΔSizeStrategy end 

"""
    WithKWargs{KW, S} <: AbstractAfterΔSizeStrategy
    WithKwargs(strategy; kwargs...)

Adds keyword argument `kwargs` to any calls to [`Δsize!`](@ref Δsize!(::Any, ::AbstractVector, ::AbstractVector)) after sizes have been determined.
"""
struct WithKwargs{KW, S} <: AbstractAfterΔSizeStrategy
    kwargs::KW
    strategy::S
end
WithKwargs(strategy; kwargs...) = WithKwargs(kwargs, strategy) 
base(s::WithKwargs) = s.strategy
fallback(s::WithKwargs) = fallback(s.strategy)

add_participants!(s::WithKwargs, vs=AbstractVertex[]) = add_participants!(base(s), vs)

"""
    AfterΔSizeCallback{F, S} <: AbstractAfterΔSizeStrategy
    AfterΔSizeCallback(cbfun::F, basestrat::S=ThrowΔSizeFailError())

Calls `cbfun(v, Δ, isnout)` for all vertices which change size after having been asked to change their sizes as a result of `basestrat`.
"""
struct AfterΔSizeCallback{F, S} <: AbstractAfterΔSizeStrategy
    cbfun::F
    base::S
end
AfterΔSizeCallback(cbfun) = AfterΔSizeCallback(cbfun, ThrowΔSizeFailError())
base(s::AfterΔSizeCallback) = s.base
add_participants!(s::AfterΔSizeCallback, vs=AbstractVertex[]) = add_participants!(base(s), vs)
fallback(s::AfterΔSizeCallback) = fallback(base(s))

"""
    logafterΔsize(printfun=nameorrepr;level=Logging.Info, base=DefaultJuMPΔSizeStrategy()) 

Return an [`AfterΔSizeCallback`](@ref) configured to log size changes with log level `level`.

For a given vertex `v`, `printfun(v)` will be used in the logged string.

Strategy `base` will be used to change sizes (e.g if [`Δsize!(logafterΔsize(base))`](@ref) is called).
"""
logafterΔsize(printfun=nameorrepr;level=Logging.Info, base=DefaultJuMPΔSizeStrategy()) = AfterΔSizeCallback(base) do v, Δ, dirlabel, ischanged
    ischanged || return
    dstr, Δstr = if dirlabel === :nout 
                        "nout", compressed_string(Δ)
                    else 
                        "nin", join(compressed_string.(Δ), ", ", " and ")
                    end
    @logmsg level "Change $dstr of $(printfun(v)) by $Δstr"
end 

"""
    validateafterΔsize(printfun=nameorrepr; base=DefaultJuMPΔSizeStrategy())

Return an [`AfterΔSizeCallback`](@ref) configured to validate that sizes (nin and nout) are consistent
after a size change and throw a `ΔSizeFailError` if validation fails.

For a given vertex `v`, `printfun(v)` will be used in the error message should the size validation fail.

Strategy `base` will be used to change sizes (e.g if [`Δsize!(validateafterΔsize(base))`](@ref) is called).
"""
validateafterΔsize(printfun=nameorrepr; base=DefaultJuMPΔSizeStrategy()) = AfterΔSizeCallback(base) do v, Δ, dirlabel, ischanged
    dirlabel === :nout && return validate_Δnout(printfun, v, Δ)
    dirlabel === :nin && return validate_Δnin(printfun, v, Δ)
end

function validate_Δnin(pf, v::AbstractVertex, Δ)
    length(Δ) == length(inputs(v)) || throw(ArgumentError("Length of Δ must be equal to number of inputs for $(pf(v))! length(Δ) = $(length(Δ)), length(inputs(v)) = $(length(inputs(v)))"))
    nout.(inputs(v)) == nin(v) || throw(ΔSizeFailError("Nin change of $(compressed_string.(Δ)) to $(pf(v)) did not result in expected size! Expected: $(nout.(inputs(v))), actual: $(nin(v))")) 
end

function validate_Δnout(pf, v::AbstractVertex, Δ)
    nin_of_outputs = unique(mapreduce(vi -> nin(vi)[inputs(vi) .== v], vcat, outputs(v), init=nout(v)))
    nin_of_outputs == [nout(v)] || throw(ΔSizeFailError("Nout change of $(compressed_string(Δ)) to $(pf(v)) resulted in size mismatch! Nin of outputs: $nin_of_outputs, nout of this: $([nout(v)])"))
end