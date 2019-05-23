
"""
AbstractMutationVertex

Base type for vertices which can be mutated
"""
abstract type AbstractMutationVertex <: AbstractVertex end

"""
    base(v::AbstractMutationVertex)

Return base vertex
"""
function base end

inputs(v::AbstractMutationVertex)  = inputs(base(v))
outputs(v::AbstractMutationVertex) = outputs(base(v))

"""
    VisitState

Memoization struct for traversal when mutating.

Remembers visitation for both forward (in) and backward (out) directions.
"""
Maybe{T} = Union{T, Missing}
struct VisitState{T}
    in::Array{AbstractMutationVertex,1}
    out::Array{AbstractMutationVertex,1}
    contexts::Dict{AbstractMutationVertex, Vector{Maybe{T}}}
end
VisitState{T}() where T = VisitState{T}([], [], OrderedDict{AbstractMutationVertex, Vector{Maybe{T}}}())
visited_in!(s::VisitState, v::AbstractMutationVertex) = push!(s.in, v)
visited_out!(s::VisitState, v::AbstractMutationVertex) = push!(s.out, v)
has_visited_in(s::VisitState, v::AbstractMutationVertex) = v in s.in
has_visited_out(s::VisitState, v::AbstractMutationVertex) = v in s.out

no_context(s::VisitState{T}) where T = isempty(s.contexts)
context!(defaultfun, s::VisitState{T}, v::AbstractMutationVertex) where T = get!(defaultfun, s.contexts, v)
delete_context!(s::VisitState{T}, v::AbstractMutationVertex) where T = delete!(s.contexts, v)
contexts(s::VisitState{T}) where T = s.contexts

"""
    OutputsVertex

Decorates an AbstractVertex with output edges.
"""
struct OutputsVertex <: AbstractVertex
    base::AbstractVertex
    outs::AbstractArray{AbstractVertex,1}
end
OutputsVertex(v::AbstractVertex) = OutputsVertex(v, AbstractVertex[])
init!(v::OutputsVertex, p::AbstractVertex) = foreach(in -> push!(outputs(in), p), inputs(v))
inputs(v::OutputsVertex) = inputs(base(v))
outputs(v::OutputsVertex) = v.outs
base(v::OutputsVertex) = v.base

## Generic helper methods

function invisit(v::AbstractMutationVertex, s::VisitState{T}) where T
    has_visited_in(s, v) && return true
    visited_in!(s, v)
    return false
end

function outvisit(v::AbstractMutationVertex, s::VisitState{T}) where T
    has_visited_out(s, v) && return true
    visited_out!(s, v)
    return false
end

function anyvisit(v::AbstractMutationVertex, s::VisitState{T}) where T
    in = invisit(v, s)
    out = outvisit(v, s)
    return in || out
end

function propagate_nin(v::AbstractMutationVertex, Δ::T; s::VisitState{T}) where T
    # Rundown of the idea here: The outputs of v might have more than one input
    # If such a vertex vi is found, the missing inputs are set to "missing" and
    # the Δ we have is put in a context for vi. Only if no input is missing
    # do we propagate to vi.
    # If we end up here though another input to vi the context will be populated
    # with the new Δ and eventually we have all the Δs
    # If not, the "if first" block in the end will insert zeroes for the missing
    # inputs and propagate anyways.
    # See testset "Transparent residual fork block" for a motivation

    first = no_context(s)

    for vi in outputs(v)
        ins = inputs(vi)
        Δs = context!(s, vi) do
            Array{Union{Missing, T},1}(missing, length(ins))
        end
        # Add Δ for each input which is the current vertex (at least and typically one)
        foreach(ind -> Δs[ind] = Δ, findall(vx -> vx == v, ins))

        any(ismissing.(Δs)) || Δnin(vi, Δs...; s=s)
    end

    if first
        for (v, ctx) in contexts(s)
            delete_context!(s, v)
            # Note: Other contexts may be "completed" as a consequence of this
            # However, if that happens the call below should just return immediately
            # due to vertex having been visited.
            # Should be safe to just add a check for this here or maybe remove
            # completed contexts in the above loop
            #TODO: missing => 0 will need to be handled sooner or later...
            Δnin(v, replace(ctx, missing=>0)..., s=s)
        end
    end
end

function propagate_nout(v::AbstractMutationVertex, Δ::T...; s::VisitState{T}=VisitState{T}()) where T
    for (Δi, vi) in zip(Δ, inputs(v))
        Δnout(vi, Δi; s=s)
    end
end

## Generic helper methods end


"""
    AbsorbVertex

Vertex which absorbs changes in nout or nin. An example of this is a vertex
which multiplies its input with an nin x nout matrix.
"""
struct AbsorbVertex <: AbstractMutationVertex
    base::AbstractVertex
    meta::VertexMeta

    function AbsorbVertex(b::OutputsVertex, meta::VertexMeta)
        this = new(b, meta)
        init!(b, this)
        return this
    end
end
AbsorbVertex(b::AbstractVertex, meta::VertexMeta) = AbsorbVertex(OutputsVertex(b), meta)
base(v::AbsorbVertex)::AbstractVertex = v.base

nin(v::AbsorbVertex) = nin(v.meta)
nout(v::AbsorbVertex) = nout(v.meta)

function Δnin(v::AbsorbVertex, Δ::T...; s::VisitState{T}=VisitState{T}()) where T
    invisit(v, s) && return
    Δnin(v.meta, Δ...)
    propagate_nout(v, Δ..., s=s)
end

function Δnout(v::AbsorbVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    outvisit(v,s) && return
    Δnout(v.meta, Δ)
    propagate_nin(v, Δ, s=s)
end


"""
    StackingVertex

Vertex which is transparent w.r.t mutation.

Size of output is sum of sizes of inputs. Examples of computations are scalar operations
(e.g add x to every element) and concatenation.
"""
struct StackingVertex <: AbstractMutationVertex
    base::AbstractVertex

    function StackingVertex(b::OutputsVertex)
        this = new(b)
        init!(b, this)
        return this
    end
end
StackingVertex(b::AbstractVertex) = StackingVertex(OutputsVertex(b))
base(v::StackingVertex)::AbstractVertex = v.base

nout(v::StackingVertex) = sum(nin(v))
nin(v::StackingVertex) = nout.(inputs(v))

function Δnin(v::StackingVertex, Δ::T...; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return
    propagate_nin(v, sum(Δ); s=s)
end

function Δnout(v::StackingVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return
    propagate_nin(v, Δ, s=s) # If there are multiple outputs they must all be updated
    insizes = nin(v)

    # We basically want a split of Δ weighted by each individual input size:
    Δs = round.(typeof(Δ), insizes .* Δ / sum(insizes))

    # However, we can't set any of the input sizes to 0
    # TODO: Add min_nout and min_nin functions to ensure the above is always
    # possible
    Δs = max.(Δs, -insizes .+ 1) #1 is the minimum input size
    Δdeficit = sum(Δs) - Δ
    for _ in 1:Δdeficit
        Δs[argmax(insizes .+ Δs)] -= 1
    end
    propagate_nout(v, Δs...; s=s)
end

"""
    InvariantVertex

Vertex which is size invariant in the sense that all inputs and outputs have the same size.

Examples of computations are scalar and element wise operations.
"""
struct InvariantVertex <: AbstractMutationVertex
    base::AbstractVertex

    function InvariantVertex(b::OutputsVertex)
        this = new(b)
        init!(b, this)
        return this
    end
end
InvariantVertex(b::AbstractVertex) = InvariantVertex(OutputsVertex(b))
base(v::InvariantVertex)::AbstractVertex = v.base

nout(v::InvariantVertex) = nin(v)[1]
nin(v::InvariantVertex) = nout.(inputs(v))

function Δnin(v::InvariantVertex, Δ::T...; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    Δprop = [Δi for Δi in unique((Δ)) if Δi != 0]
    @assert length(Δprop) == 1 "Change must be invariant!"

    propagate_nin(v, Δprop...; s=s)
    propagate_nout(v, repeat(Δprop, length(inputs(v)))...; s=s)
end

function Δnout(v::InvariantVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    propagate_nin(v, Δ, s=s)
    propagate_nout(v, fill(Δ, length(inputs(v)))...; s=s)
end
