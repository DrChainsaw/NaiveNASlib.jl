
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
struct VisitState
    in::Array{AbstractMutationVertex,1}
    out::Array{AbstractMutationVertex,1}
end
VisitState() = VisitState([], [])
visited_in!(s::VisitState, v::AbstractMutationVertex) = push!(s.in, v)
visited_out!(s::VisitState, v::AbstractMutationVertex) = push!(s.out, v)
has_visited_in(s::VisitState, v::AbstractMutationVertex) = v in s.in
has_visited_out(s::VisitState, v::AbstractMutationVertex) = v in s.out

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
inputs(v::OutputsVertex) = inputs(v.base)
outputs(v::OutputsVertex) = v.outs


function invisit(v::AbstractMutationVertex, s::VisitState)
    has_visited_in(s, v) && return true
    visited_in!(s, v)
    return false
end

function outvisit(v::AbstractMutationVertex, s::VisitState)
    has_visited_out(s, v) && return true
    visited_out!(s, v)
    return false
end

function anyvisit(v::AbstractMutationVertex, s::VisitState)
    in = invisit(v, s)
    out = outvisit(v, s)
    return in || out
end

function propagate_nin(v::AbstractMutationVertex, Δ::Integer; s::VisitState)
    for vi in outputs(v)
        ins = inputs(vi)
        Δs = zeros(typeof(Δ), length(ins))
        Δs[findall(vx -> vx == v, ins)] .= Δ
        Δnin(vi, Δs...; s=s)
    end
end

function propagate_nout(v::AbstractMutationVertex, Δ::Integer...; s::VisitState=VisitState())
    for (Δi, vi) in zip(Δ, inputs(v))
        Δnout(vi, Δi; s=s)
    end
end

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

function Δnin(v::AbsorbVertex, Δ::Integer...; s::VisitState=VisitState())
    invisit(v, s) && return
    Δnin(v.meta, Δ...)
    propagate_nout(v, Δ...; s=s)
end

function Δnout(v::AbsorbVertex, Δ::Integer; s::VisitState=VisitState())
    (outvisit(v,s) || Δ == 0) && return
    Δnout(v.meta, Δ)
    propagate_nin(v, Δ; s=s)
end


"""
    TransparentVertex

Vertex which is tranpsparent w.r.t mutation.

Size of output is sum of sizes of inputs. Examples of computations are scalar operations
(e.g add x to every element) and concatenation.
"""
struct TransparentVertex <: AbstractMutationVertex
    base::AbstractVertex

    function TransparentVertex(b::OutputsVertex)
        this = new(b)
        init!(b, this)
        return this
    end
end
TransparentVertex(b::AbstractVertex) = TransparentVertex(OutputsVertex(b))
base(v::TransparentVertex)::AbstractVertex = v.base

nout(v::TransparentVertex) = sum(nin(v))
nin(v::TransparentVertex) = nout.(inputs(v))

function Δnin(v::TransparentVertex, Δ::Integer...; s::VisitState=VisitState())
    anyvisit(v, s) && return
    propagate_nin(v, sum(Δ); s=s)
end

function Δnout(v::TransparentVertex, Δ::Integer; s::VisitState=VisitState())
    (anyvisit(v, s) || Δ == 0) && return
    propagate_nin(v, Δ; s=s) # If there are multiple outputs they must all be updated
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
