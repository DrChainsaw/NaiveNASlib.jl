

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
LogΔSizeExec(msg::String)
LogΔSizeExec(level::Logging.LogLevel, msg::String)
LogΔSizeExec(level::Logging.LogLevel, msg::String, andthen::AbstractJuMPΔSizeStrategy)

Logs `msg` at log level `level`, then executes `AbstractJuMPΔSizeStrategy andthen`.
"""
struct LogΔSizeExec <: AbstractJuMPΔSizeStrategy
level::LogLevel
msg::String
andthen::AbstractJuMPΔSizeStrategy
end
LogΔSizeExec(msg::String) = LogΔSizeExec(Logging.Info, msg)
LogΔSizeExec(level::LogLevel, msg::String) = LogΔSizeExec(level, msg, ΔSizeFailNoOp())
LogSelectionFallback(nextstr, andthen; level=Logging.Warn) =LogΔSizeExec(level, v -> "Selection for vertex $(name(v)) failed! $nextstr", andthen)


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
vertex::V
Δ::Int
fallback::F
end
ΔNoutExact(v::AbstractVertex, Δ::Integer) = ΔNout{Exact}(v, Δ, LogΔSizeExec(Logging.Warn, "Could not change nout of $v by $(Δ)! Relaxing constraints...", ΔNoutRelaxed(v, Δ)))
ΔNoutRelaxed(v::AbstractVertex, Δ::Integer) = ΔNout{Relaxed}(v, Δ, ΔSizeFailError("Could not change nout of $v by $(Δ)!!"))
ΔNout{T}(v::V, Δ::Integer, f::F) where {V, T, F} = ΔNout{T, V, F}(v, Int(Δ), f)
fallback(s::ΔNout) = s.fallback

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
struct ΔNin{T} <: AbstractJuMPΔSizeStrategy
vertices::Vector{<:AbstractVertex}
Δs::Vector{Int}
fallback::AbstractJuMPΔSizeStrategy
function ΔNin{T}(v, Δs, fallback) where T
    @assert size(Δs) == size(inputs(v)) "Must supply same number of Δs as v has inputs! Got $Δs for $v."
    inds = .!ismissing.(Δs)
    new(inputs(v)[inds], Δs[inds], fallback)
end
end
ΔNinExact(v::AbstractVertex, Δs::Vector{<:Maybe{Int}}) = ΔNin{Exact}(v, Δs, LogΔSizeExec(Logging.Warn, "Could not change nin of $v by $(join(Δs, ", "))! Relaxing constraints...", ΔNinRelaxed(v, Δs)))
ΔNinExact(v::AbstractVertex, Δ::Integer) = ΔNin{Exact}(v, [Δ])
ΔNinRelaxed(v::AbstractVertex, Δs::Vector{<:Maybe{Int}}) = ΔNin{Relaxed}(v, Δs, ΔSizeFailError("Could not change nin of $vertex by $(join(Δs, ", "))!!"))
ΔNinRelaxed(v::AbstractVertex, Δ::Integer) = ΔNin{Relaxed}(v, [Δ])
fallback(s::ΔNin) = s.fallback


"""
AlignNinToNout <: AbstractJuMPΔSizeStrategy
AlignNinToNout(vstrat=DefaultJuMPΔSizeStrategy())

Adds variables and constraints for `nin(vi) == nout.(inputs(vi))`.

If it fails, the operation will be retried with the `fallback` strategy (default `ΔSizeFailError`).
"""
struct AlignNinToNout <: AbstractJuMPΔSizeStrategy
nindict::Dict{AbstractVertex, Vector{JuMP.VariableRef}}
vstrat::AbstractJuMPΔSizeStrategy
fallback::AbstractJuMPΔSizeStrategy
end
AlignNinToNout(;vstrat=DefaultJuMPΔSizeStrategy(), fallback=ΔSizeFailError("Failed to align Nin to Nout!!")) = AlignNinToNout(vstrat, fallback)
AlignNinToNout(vstrat, fallback) = AlignNinToNout(Dict{AbstractVertex, JuMP.VariableRef}(), vstrat, fallback)
fallback(s::AlignNinToNout) = s.fallback

"""
AlignNinToNoutVertices <: AbstractJuMPΔSizeStrategy
AlignNinToNoutVertices(vin, vout, inds::Integer...;vstrat=AlignNinToNout(), fallback=ΔSizeFailError())
AlignNinToNoutVertices(vin, vout, inds::AbstractArray{<:Integer},vstrat=AlignNinToNout(), fallback=ΔSizeFailError())

Same as [`AlignNinToNout`](@ref) with an additional constraint that `nin(s.vin)[s.ininds] == nout(s.vout)` where `s` is a `AlignNinToNoutVertices`.

Useful in the context of removing vertices and/or edges.

If it fails, the operation will be retried with the `fallback` strategy (default `ΔSizeFailError`).
"""
struct AlignNinToNoutVertices <: AbstractJuMPΔSizeStrategy
vin::AbstractVertex
vout::AbstractVertex
ininds::AbstractArray{<:Integer}
vstrat::AlignNinToNout
fallback::AbstractJuMPΔSizeStrategy
end
AlignNinToNoutVertices(vin, vout, inds::Integer...;vstrat=AlignNinToNout(), fallback=failToAlign(vin, vout)) = AlignNinToNoutVertices(vin, vout, collect(inds), vstrat, fallback)
AlignNinToNoutVertices(vin, vout, inds::AbstractArray{<:Integer}, vstrat::AlignNinToNout, fallback::AbstractJuMPΔSizeStrategy=failToAlign(vin,vout)) = AlignNinToNoutVertices(vin, vout, inds, vstrat, fallback)
AlignNinToNoutVertices(vin, vout, inds::AbstractArray{<:Integer}, innerstrat::AbstractJuMPΔSizeStrategy, fallback=failtoalign(vin, vout)) = AlignNinToNoutVertices(vin, vout, inds, AlignNinToNout(vstrat=innerstrat), fallback)
fallback(s::AlignNinToNoutVertices) = s.fallback

failtoalign(vin, vout) = ΔSizeFailError("Could not align nout of $vin to nin of $(vout)!!")

"""
    SelectDirection <: AbstractΔSizeStrategy
    SelectDirection()
    SelectDirection(s::AbstractΔSizeStrategy)

Select indices for a vertex using `AbstractΔSizeStrategy s` (default `OutSelect{Exact}`) in only the direction(s) in which the vertex has changed size.

Intended use it to reduce the number of constraints for a `AbstractJuMPΔSizeStrategy` as only the parts of the graph which are changed will be considered.
"""
struct SelectDirection{S <: AbstractΔSizeStrategy} <: AbstractΔSizeStrategy
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
struct TruncateInIndsToValid{S <: AbstractΔSizeStrategy} <: AbstractΔSizeStrategy
    strategy::S
end
TruncateInIndsToValid() = TruncateInIndsToValid(OutSelectExact())
