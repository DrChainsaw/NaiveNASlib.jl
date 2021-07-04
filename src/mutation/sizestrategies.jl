

"""
    AbstractΔSizeStrategy

Abstract base type for strategies for how to change the size.

Only used as a transition until JuMP approach has been fully verified.
"""
abstract type AbstractΔSizeStrategy end

"""
OnlyFor <: AbstractΔSizeStrategy

Change size only for the provided vertex.
"""
struct OnlyFor <: AbstractΔSizeStrategy end

"""
    AbstractJuMPΔSizeStrategy <: AbstractΔSizeStrategy

Abstract type for strategies to change or align the sizes of vertices using JuMP.
"""
abstract type AbstractJuMPΔSizeStrategy <: AbstractΔSizeStrategy end

"""
    DecoratingJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy

Abstract type for AbstractJuMPΔSizeStrategies which wants to delegate some parts of the problem formulation to another strategy.

More concretely: If `s` is a `DecoratingJuMPΔSizeStrategy` then `base(s)` will be used unless explicitly stated through dispatch.
"""
abstract type DecoratingJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy end

"""
    ΔSizeFail <: Exception

Size change could not be solved.
"""
struct ΔSizeFailError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ΔSizeFailError) = print(io, "ΔSizeFailError: ", e.msg)

"""
    ThrowΔSizeFailError <: AbstractJuMPΔSizeStrategy
    ThrowΔSizeFailError(msg::String)

Throws an `ErrorException` with message `msg`.
"""
struct ThrowΔSizeFailError{F} <: AbstractJuMPΔSizeStrategy
    msgfun::F
end
ThrowΔSizeFailError(msg::AbstractString) = ThrowΔSizeFailError(vs -> msg)
ThrowΔSizeFailError() = ThrowΔSizeFailError(vs -> string("Could not change size of vertices ", join(nameorrepr.(vs), ", ", " and "),"!")) 

"""
    ΔSizeFailNoOp <: AbstractJuMPΔSizeStrategy
    ΔSizeFailNoOp()

Does not perform any action.
"""
struct ΔSizeFailNoOp <: AbstractJuMPΔSizeStrategy end

"""
    LogΔSizeExec <: AbstractJuMPΔSizeStrategy
    LogΔSizeExec(msgfun)
    LogΔSizeExec(msgfun, level::Logging.LogLevel)
    LogΔSizeExec(msgfun, level::Logging.LogLevel, andthen::AbstractJuMPΔSizeStrategy)

Logs `msgfun(v)` at log level `level`, then executes `AbstractJuMPΔSizeStrategy andthen` for vertex `v`.
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
fallback(s::LogΔSizeExec) = base(s)


"""
    DefaultJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy

Default strategy intended to be used when adding some extra constraints or objectives to a model on top of the default.
"""
struct DefaultJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy end

struct Exact end
struct Relaxed end

"""
    ΔNout <: AbstractJuMPΔSizeStrategy
    ΔNout{T, V, F}(Δs::Dict{V, Int}, fallback::F)

Strategy for changing nout of vertices.
    
For each key-value pair in `v, Δ` in `Δs`, change `nout(v)` by `Δ`, i.e new size is `nout(v) + Δ`.

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

generic_Δnin_docstring_examples(fname::String; kwargs...) = generic_Δnin_docstring_examples(args ->  "$fname($args)"; kwargs...)
generic_Δnin_docstring_examples(fgen; footer = "```") = generic_Δnout_docstring_examples(fgen; footer="") * """

# Assume v1 has three inputs and v2 has two inputs instead
julia> NaiveNASlib.inputs(v::DummyVertex) = v === v1 ? [v,v,v] : [v,v]

julia> $(fgen("v1, 3, missing, 2"));

julia> $(fgen("v1 => (3, missing, 2), v2 => (-2, 0)"));
""" * footer

generic_Δnout_docstring_examples(fname::String; kwargs...) = generic_Δnout_docstring_examples(args -> "$fname($args)"; kwargs...)
generic_Δnout_docstring_examples(fgen; footer = "```") = """
### Examples
```julia-repl
julia> struct DummyVertex <: NaiveNASlib.AbstractVertex id::Int end

julia> NaiveNASlib.inputs(v::DummyVertex) = [v]

julia> v1,v2 = DummyVertex(1), DummyVertex(2)

julia> $(fgen("v1, 3"));

julia> $(fgen("v1 => 3, v2 => -2"));

julia> $(fgen("Dict(v1 => 3, v2 => -2)"));
""" * footer

"""
    ΔNout{T}(args...;fallback::F)

Return a `ΔNout{T, V, F}` created from `args` where `args` can be either a `Dict` or arguments used to create a `Dict`
(e.g. a set of `Pair`s or a generator).

Also helps creating a `Dict{V, Int}` from e.g. a `Dict{Any, Any}` as a convenience.

$(generic_Δnout_docstring_examples(args -> "ΔNout{Exact}($args; fallback=ThrowΔSizeFailError()"))
"""
ΔNout{T}(v::AbstractVertex, Δ::Integer; fallback) where T = ΔNout{T}(v=>Int(Δ); fallback)
ΔNout{T}(args...; fallback) where T = ΔNout{T}(Dict(args...); fallback)
ΔNout{T}(Δs::AbstractDict; fallback) where T = ΔNout{T}(Dict(k=>Int(v) for (k,v) in Δs); fallback)
ΔNout{T}(Δs::AbstractDict{V, Int}; fallback::F) where {T,V,F} = ΔNout{T,V,F}(Dict(Δs), fallback)

"""
    ΔNoutExact(args...; fallback)

Return a ΔNout{Exact} with `fallback` set to give a warning and then try again with `ΔNoutRelaxed(args...)`.

Accepts either a `Dict` or arguments used to create a `Dict` (e.g. a set of `Pair`s or a generator).

$(generic_Δnout_docstring_examples("ΔNoutExact"))
"""
ΔNoutExact(args...; fallback=default_noutfallback(ΔNoutRelaxed,"nout", args)) = ΔNout{Exact}(args...;fallback)

"""
    ΔNoutRelaxed(args...;fallback)

Return a ΔNout{Relaxed} with `fallback` set to `ThrowΔSizeFailError`.

Accepts either a `Dict` or arguments used to create a `Dict` (e.g. a set of `Pair`s or a generator).

$(generic_Δnout_docstring_examples("ΔNoutRelaxed"))
"""
ΔNoutRelaxed(args...;fallback=default_noutfallback("nout", args)) = ΔNout{Relaxed}(args...;fallback)

Δnout_err_info(v, Δ::Union{<:Maybe{Int}, <:Pair}...) = "$(nameorrepr(v)) by $(join(first.(Δ), ", ", " and "))"
Δnout_err_info(v, Δ::Tuple) = Δnout_err_info(v, Δ...)
Δnout_err_info(ps::Pair...) = join(map(p ->Δnout_err_info(p...), ps), ", ", " and ")
Δnout_err_info(d::AbstractDict) = join((Δnout_err_info(k, v) for (k,v) in d), ", ", " and ")

function default_noutfallback(nextfallback, dirstr, args) 
    msgfun = v -> string("Could not change ", dirstr, " of ", Δnout_err_info(args...), "! Relaxing constraints...")
    return LogΔSizeExec(msgfun, Logging.Warn, nextfallback(args...))
end

default_noutfallback(dirstr, args) = ThrowΔSizeFailError(() -> string("Could not change ", dirstr, " of ", Δnout_err_info(args...), "!!"))

"""
    ΔNinExact(args...; fallback)

Return a `ΔNout{Exact}` configured to set `nout` of the inputs to vertices in `args` to the given sizes.

For vertices with more than one input, the size change must be expressed as a tuple with one element per input. 
Use `missing` to indicate that no special treatment is needed for an input.

Accepts either a `Dict` or arguments used to create a `Dict` (e.g. a set of `Pair`s or a generator).

By default, `fallback` is set to give a warning and then try again with `ΔNinRelaxed(args...)`.

$(generic_Δnin_docstring_examples("ΔNinExact"))
"""
ΔNinExact(args...; fallback=default_noutfallback(ΔNinRelaxed, "nin", args)) = ΔNout{Exact}(Δnin2Δnout(args...); fallback)

"""
    ΔNinRelaxed(args...; fallback)

Same as [`ΔNinExact`](@ref) except `Exact` is replaced by `Relaxed` and `fallback` set to `ThrowΔSizeFailError` by default.

$(generic_Δnin_docstring_examples("ΔNinRelaxed"))
"""
ΔNinRelaxed(args...; fallback=default_noutfallback("nout", args)) = ΔNout{Relaxed}(Δnin2Δnout(args...); fallback)

Δnin2Δnout(d::AbstractDict) = reduce(merge!, (Δnin2Δnout(k,v) for (k,v) in d); init=Dict())
Δnin2Δnout(ps::Pair{<:AbstractVertex}...) = mapreduce(p -> Δnin2Δnout(p...), merge!, ps; init=Dict()) 
Δnin2Δnout(v::AbstractVertex, Δs) = Δnin2Δnout(v, tuple(Δs))
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

"""
    relaxed(Δ)

Return `Δ => Relaxed()` which indicates that `Δ` shall be relaxed when changing size.

See [`Δnout`](@ref) and [`Δnin`](@ref).
"""
relaxed(Δ::Integer) = Δ => Relaxed()
relaxed(Δs::Tuple{Vararg{<:Maybe{Int}}}) = Δs => Relaxed()

"""
    ΔNout(args...)

Splits `args` into relaxed and exact size changes and creates the appropriate strategy (one of `ΔNout{Exact}, ΔNout{Relaxed} or ΔNoutMix`).

$(generic_Δnout_docstring_examples("ΔNout"; footer=""))
julia> ΔNout(v1, relaxed(2));

julia> ΔNout(v1 => relaxed(2), v2 => -1)
``` 
"""
function ΔNout(args...) 
    exact, relaxed = split_exact_relaxed(args)
    return ΔNout(exact, relaxed)
end

"""
    ΔNin(args...)

Splits `args` into relaxed and exact size changes and creates the appropriate strategy (one of `ΔNout{Exact}, ΔNout{Relaxed} or ΔNoutMix`).

$(generic_Δnin_docstring_examples("ΔNin"; footer=""))
julia> ΔNin(v1, relaxed(3), missing, 2);

julia> ΔNin(v1 => (relaxed(3), missing, 2), v2 => relaxed(-2, 0))
``` 
"""
function ΔNin(args...) 
    exact,relaxed = split_exact_relaxed(Δnin2Δnout(args...)) # To support mixed cases, e.g. Δnin(v => (2, relaxed(4)))
    fallback = default_noutfallback((args...) -> ΔNoutRelaxed(merge(exact, relaxed); fallback=default_noutfallback("nin", args)), "nin", args)
    return ΔNout(exact, relaxed, fallback)
end

function ΔNout(exact::Dict, relaxed::Dict, fb = default_noutfallback(ΔNoutRelaxed, "nout", merge(exact,relaxed)))
    isempty(relaxed) && return ΔNoutExact(exact;fallback=fb)
    isempty(exact) && return ΔNoutRelaxed(relaxed; fallback=fallback(fb))
    return ΔNoutMix(ΔNoutExact(exact;fallback=fb), ΔNoutRelaxed(relaxed;fallback=fb), fb)
end

split_exact_relaxed(p::Tuple{AbstractVertex, Any}) = split_exact_relaxed(tuple(Pair(p...)))
split_exact_relaxed(d::Tuple{Dict}) = split_exact_relaxed(first(d))
function split_exact_relaxed(args::Union{AbstractDict{V}, Tuple{Vararg{Pair{V}}}}) where V
    exact=Dict{V, Int}()
    relaxed=Dict{V, Int}() 
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
    AlignNinToNout <: DecoratingJuMPΔSizeStrategy
    AlignNinToNout(vstrat=DefaultJuMPΔSizeStrategy())

Adds variables and constraints for `nin(vi) == nout.(inputs(vi))`.

Generally useful when doing structural changes such are removing/adding vertices or edges.

If it fails, the operation will be retried with the `fallback` strategy (default `ThrowΔSizeFailError`).
"""
struct AlignNinToNout{S, F} <: DecoratingJuMPΔSizeStrategy
    nindict::Dict{AbstractVertex, Vector{JuMP.VariableRef}}
    vstrat::S
    fallback::F
end
AlignNinToNout(;vstrat=DefaultJuMPΔSizeStrategy(), fallback=ThrowΔSizeFailError("Failed to align Nin to Nout!!")) = AlignNinToNout(vstrat, fallback)
AlignNinToNout(vstrat, fallback) = AlignNinToNout(Dict{AbstractVertex, Vector{JuMP.VariableRef}}(), vstrat, fallback)
fallback(s::AlignNinToNout) = s.fallback
base(s::AlignNinToNout) = s.vstrat

"""
    AlignNinToNoutVertices <: AbstractJuMPΔSizeStrategy
    AlignNinToNoutVertices(vin, vout, inds::Integer...;vstrat=AlignNinToNout(), fallback=ThrowΔSizeFailError())
    AlignNinToNoutVertices(vin, vout, inds::AbstractArray{<:Integer},vstrat=AlignNinToNout(), fallback=ThrowΔSizeFailError())

Same as [`AlignNinToNout`](@ref) with an additional constraint that `nin(s.vin)[s.ininds] == nout(s.vout)` where `s` is a `AlignNinToNoutVertices`.

Useful in the context of removing vertices and/or edges.

If it fails, the operation will be retried with the `fallback` strategy (default `ThrowΔSizeFailError`).
"""
struct AlignNinToNoutVertices{V1,V2,F} <: DecoratingJuMPΔSizeStrategy
    vin::V1
    vout::V2
    ininds::Vector{Int}
    vstrat::AlignNinToNout
    fallback::F
end
AlignNinToNoutVertices(vin, vout, inds::Integer...;vstrat=AlignNinToNout(), fallback=failToAlign(vin, vout)) = AlignNinToNoutVertices(vin, vout, collect(inds), vstrat, fallback)
AlignNinToNoutVertices(vin, vout, inds::AbstractArray{<:Integer}, vstrat::AlignNinToNout, fallback::AbstractJuMPΔSizeStrategy=failToAlign(vin,vout)) = AlignNinToNoutVertices(vin, vout, inds, vstrat, fallback)
AlignNinToNoutVertices(vin, vout, inds::AbstractArray{<:Integer}, innerstrat::AbstractJuMPΔSizeStrategy, fallback=failtoalign(vin, vout)) = AlignNinToNoutVertices(vin, vout, inds, AlignNinToNout(vstrat=innerstrat), fallback)
fallback(s::AlignNinToNoutVertices) = s.fallback
base(s::AlignNinToNoutVertices) = s.vstrat

failtoalign(vin, vout) = ThrowΔSizeFailError(vs -> string("Could not align nout of ", nameorrepr(vin), " to nin of ", nameorrepr(vout), "!!"))

"""
    SelectDirection <: AbstractΔSizeStrategy
    SelectDirection()
    SelectDirection(s::AbstractΔSizeStrategy)

Select indices for a vertex using `AbstractΔSizeStrategy s` (default `DefaultJuMPSelectionStrategy`) in only the direction(s) in which the vertex has changed size.

Intended use it to reduce the number of constraints for a `AbstractJuMPΔSizeStrategy` as only the parts of the graph which are changed will be considered.
"""
struct SelectDirection{S} <: AbstractΔSizeStrategy
    strategy::S
end
SelectDirection() = SelectDirection(DefaultJuMPΔSizeStrategy())

"""
    TruncateInIndsToValid{S} <: AbstractΔSizeStrategy
    TruncateInIndsToValid()
    TruncateInIndsToValid(s::S)

Ensures that all selected input indices are within range of existing input indices after applying `s` 
(default `DefaultJuMPSelectionStrategy`).

Not needed in normal cases, but certain structural mutations (e.g create_edge!) may cause this to happen 
due to how constraints are (not) created when original sizes do not align in conjunction with how result of 
selection is interpreted.

An example of when it is needed is when adding an edge from vertex `vi` to an invariant vertex `vo` where 
`nout(vi) > nout(vo)`.In this case it is expected that the result of the optimization is that the indices 
`1:nout(vi)` of `vi` shall be kept. However, this will propagate to `vo` which will be instructed to keep 
indices it does not have. With this strategy, all indices which are larger than `nout(vo)` will be replaced
by `-1` (which indicates that new parameters shall be created) 

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

"""
    WithValueFun{F, S} <: AbstractΔSizeStrategy
    WithValueFun(valuefun::F, strategy::S)

Applies neuron indices selection with `strategy` and using `valuefun` to compute the value of neurons indices.

Note that `valuefun` will override any value function supplied in function call. Thus it is possible use 
`WithValueFun` to change value function e.g. when switching to a fallback strategy.
"""
struct WithValueFun{F, S} <: AbstractΔSizeStrategy
    valuefun::F
    strategy::S
end
WithValueFun(f) = s -> WithValueFun(f ,s) 
base(s::WithValueFun) = s.strategy
fallback(s::WithValueFun) = fallback(s.strategy)
