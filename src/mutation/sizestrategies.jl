

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
ΔSizeFailError <: AbstractJuMPΔSizeStrategy
ΔSizeFailError(msg::String)

Throws an `ErrorException` with message `msg`.
"""
struct ΔSizeFailError <: AbstractJuMPΔSizeStrategy
    msg::String
end

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
LogΔSizeExec(msg::String, level::LogLevel=Logging.Info, andthen=ΔSizeFailNoop()) = LogΔSizeExec(v -> msg, level, andthen)
LogSelectionFallback(nextstr, andthen; level=Logging.Warn) = LogΔSizeExec(v -> "Size change for vertex $(name(v)) failed! $nextstr", level, andthen)
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
ΔNout{T} <: AbstractJuMPΔSizeStrategy
ΔNout{T}(vertex::AbstractVertex, Δ::Integer)
ΔNoutExact(vertex::AbstractVertex, Δ::Integer, fallback::AbstractJuMPΔSizeStrategy)
ΔNoutRelaxed(vertex::AbstractVertex, Δ::Integer, fallback::AbstractJuMPΔSizeStrategy)

Strategy for changing nout of `vertex` by `Δ`, i.e new size is `nout(vertex) + Δ`.

If `T == Exact`, size change will be added as a constraint to the model which means that the operation will fail if it is not possible to change `nout(vertex)` by exactly `Δ`. If the operation fails, it will be retried with the `fallback` strategy (default `ΔNoutRelaxed`).

If `T == Relaxed`, size change will be added as an objective to the model which means that `nout(vertex)` might not change by exactly `Δ`. In addition, a constraint that `nout(vertex)` must change is also added.

If the operation fails, it will be retried with the `fallback` strategy (default `ΔNout{Relaxed}` if `T==Exact` and `ΔSizeFailError` if `T==Relaxed`).
"""
struct ΔNout{T, V, F} <: AbstractJuMPΔSizeStrategy
    Δs::Dict{V, Int}
    fallback::F
end
ΔNoutExact(args...; fallback=default_noutfallback(ΔNoutRelaxed,"nout", args)) = ΔNout{Exact}(args...;fallback)
ΔNoutRelaxed(args...;fallback=default_noutfallback("nout", args)) = ΔNout{Relaxed}(args...;fallback)
ΔNout{T}(v::AbstractVertex, Δ::Integer; fallback) where T = ΔNout{T}(v=>Int(Δ); fallback)
ΔNout{T}(args...; fallback) where T = ΔNout{T}(Dict(args...); fallback)
function ΔNout{T}(Δs::AbstractDict; fallback) where T 
    @assert all(v -> v isa Int, values(Δs)) string("All values must be of type Int. Got ", values(Δs))
    return ΔNout{T}(Dict(k=>v for (k,v) in Δs); fallback)
end
ΔNout{T}(Δs::AbstractDict{V, Int}; fallback) where {T,V} = ΔNout{T,V,typeof(fallback)}(Dict(Δs), fallback)
fallback(s::ΔNout) = s.fallback

Δnout_err_info(v, Δ::Union{<:Maybe{Int}, <:Pair}...) = "$v by $(join(first(Δ), ", ", " and "))"
Δnout_err_info(v, Δ::Tuple) = Δnout_err_info(v, Δ...)
Δnout_err_info(ps::Pair...) = join(map(p ->Δnout_err_info(p...), ps), ", ", " and ")
Δnout_err_info(d::AbstractDict) = join((Δnout_err_info(k, v) for (k,v) in d), ", ", " and ")

function default_noutfallback(nextfallback, dirstr, args) 
    LogΔSizeExec(v -> string("Could not change ", dirstr, " of ", Δnout_err_info(args...), "! Relaxing constraints..."), Logging.Warn, nextfallback(args...))
end

default_noutfallback(dirstr, args) = ΔSizeFailError(string("Could not change ", dirstr, " of ", Δnout_err_info(args...), "!!"))


"""
ΔNin{T} <: AbstractJuMPΔSizeStrategy
ΔNin{T}(vertex::AbstractVertex, Δs::Vector{Maybe{Int}}, fallback::AbstractJuMPΔSizeStrategy)
ΔNinExact(vertex::AbstractVertex, Δs::Vector{Maybe{Int}})
ΔNinRelaxed(vertex::AbstractVertex, Δs::Vector{Maybe{Int}})

Strategy for changing nin of `vertex` by `Δs`, i.e new size is `nin(vertex) .+ Δs`. Note that `Δs` must have the same number of elements as `nin(vertex)`.

Use `missing` to indicate "no change required" as 0 will be interpreted as "must not change".

If `T == Exact`, size change will be added as a constraint to the model which means that the operation will fail if it is not possible to change `nin(vertex)` by exactly `Δs`.

If `T == Relaxed`, size change will be added as an objective to the model which means that `nin(vertex)` might not change by exactly `Δs`. In addition, a constraint that `nin(vertex)` must change is also added.

If the operation fails, it will be retried with the `fallback` strategy (default `ΔNin{Relaxed}` if `T==Exact` and `ΔSizeFailError` if `T==Relaxed`).
"""
ΔNinExact(args...; fallback=default_noutfallback(ΔNinRelaxed, "nin", args)) = ΔNin(Exact(), args...; fallback)
ΔNinRelaxed(args...; fallback=default_noutfallback("nout", args)) = ΔNin(Relaxed(), args...;fallback)
ΔNin(::T, args...; fallback) where T<:Union{Relaxed, Exact} = ΔNout{T}(Δnin2Δnout(args...); fallback)

Δnin2Δnout(d::AbstractDict) = reduce(merge!, (Δnin2Δnout(k,v) for (k,v) in d); init=Dict())
Δnin2Δnout(ps::Pair{<:AbstractVertex}...) = mapreduce(p -> Δnin2Δnout(p...), merge!, ps; init=Dict()) 
Δnin2Δnout(v::AbstractVertex, Δs) = Δnin2Δnout(v, tuple(Δs))
Δnin2Δnout(v::AbstractVertex, Δs::Pair{<:Tuple, <:Union{Relaxed, Exact}}) = Dict(k => (v => last(Δs)) for (k,v) in Δnin2Δnout(v, first(Δs))) 
function Δnin2Δnout(v::AbstractVertex, Δs::Tuple)
    @assert length(Δs) == length(inputs(v)) "Must supply same number of Δs as v has inputs! Got $Δs for $v."
    inds = findall(!ismissing, Δs)
    return Dict(inputs(v)[i] => Δs[i] for i in inds)
end

struct ΔNoutMix{VE, VR, F} <: AbstractJuMPΔSizeStrategy
    exact::ΔNout{Exact, VE, F}
    relax::ΔNout{Relaxed, VR, F}
    fallback::F
end
fallback(s::ΔNoutMix) = s.fallback

relaxed(i::Int) = i => Relaxed()
relaxed(t::Tuple{Vararg{Int}}) = t => Relaxed()

function ΔNin(args...) 
    exact,relaxed = split_exact_relaxed(Δnin2Δnout(args...)) # To support mixed cases, e.g. Δnin(v => (2, relaxed(4)))
    fallback = default_noutfallback((args...) -> ΔNoutRelaxed(merge(exact, relaxed); fallback=default_noutfallback("nin", args)), "nin", args)
    return ΔNout(exact, relaxed, fallback)
end
function ΔNout(args...) 
    exact, relaxed = split_exact_relaxed(args)
    return ΔNout(exact, relaxed)
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
AlignNinToNout <: AbstractJuMPΔSizeStrategy
AlignNinToNout(vstrat=DefaultJuMPΔSizeStrategy())

Adds variables and constraints for `nin(vi) == nout.(inputs(vi))`.

If it fails, the operation will be retried with the `fallback` strategy (default `ΔSizeFailError`).
"""
struct AlignNinToNout{S, F} <: DecoratingJuMPΔSizeStrategy
    nindict::Dict{AbstractVertex, Vector{JuMP.VariableRef}}
    vstrat::S
    fallback::F
end
AlignNinToNout(;vstrat=DefaultJuMPΔSizeStrategy(), fallback=ΔSizeFailError("Failed to align Nin to Nout!!")) = AlignNinToNout(vstrat, fallback)
AlignNinToNout(vstrat, fallback) = AlignNinToNout(Dict{AbstractVertex, Vector{JuMP.VariableRef}}(), vstrat, fallback)
fallback(s::AlignNinToNout) = s.fallback
base(s::AlignNinToNout) = s.vstrat

"""
AlignNinToNoutVertices <: AbstractJuMPΔSizeStrategy
AlignNinToNoutVertices(vin, vout, inds::Integer...;vstrat=AlignNinToNout(), fallback=ΔSizeFailError())
AlignNinToNoutVertices(vin, vout, inds::AbstractArray{<:Integer},vstrat=AlignNinToNout(), fallback=ΔSizeFailError())

Same as [`AlignNinToNout`](@ref) with an additional constraint that `nin(s.vin)[s.ininds] == nout(s.vout)` where `s` is a `AlignNinToNoutVertices`.

Useful in the context of removing vertices and/or edges.

If it fails, the operation will be retried with the `fallback` strategy (default `ΔSizeFailError`).
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

failtoalign(vin, vout) = ΔSizeFailError("Could not align nout of $vin to nin of $(vout)!!")

"""
    SelectDirection <: AbstractΔSizeStrategy
    SelectDirection()
    SelectDirection(s::AbstractΔSizeStrategy)

Select indices for a vertex using `AbstractΔSizeStrategy s` (default `OutSelect{Exact}`) in only the direction(s) in which the vertex has changed size.

Intended use it to reduce the number of constraints for a `AbstractJuMPΔSizeStrategy` as only the parts of the graph which are changed will be considered.
"""
struct SelectDirection{S} <: AbstractΔSizeStrategy
    strategy::S
end
SelectDirection() = SelectDirection(OutSelectExact())

"""
    TruncateInIndsToValid{S} <: AbstractΔSizeStrategy
    TruncateInIndsToValid()
    TruncateInIndsToValid(s::S)

Ensures that all selected input indices are within range of existing input indices after applying `s` (default `OutSelectExact`).

Not needed in normal cases, but certain structural mutations (e.g create_edge!) may cause this to happen due to how constraints are (not) created when original sizes do not align in conjunction with how result of selection is interpreted.

While this may be considered a flaw in the output selection procedure, it is rare enough so that in most cases when it happens it is the result of a user error or lower level bug. Therefore this strategy is left optional to be used only in cases when mismatches are expected.
"""
struct TruncateInIndsToValid{S} <: AbstractΔSizeStrategy
    strategy::S
end
TruncateInIndsToValid() = TruncateInIndsToValid(DefaultJuMPΔSizeStrategy())
