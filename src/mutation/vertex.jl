
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

Memo struct for traversal when mutating.

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

function Δnin(v::AbsorbVertex, Δ::Integer...; s::VisitState=VisitState())
    has_visited_in(s, v) && return

    visited_in!(s, v)
    Δnin(v.meta, Δ...)

    for (Δi, vi) in zip(Δ, inputs(v))
        if Δi > 0
            Δnout(vi, Δi; s=s)
        end
    end
end

function Δnout(v::AbsorbVertex, Δ::Integer; s::VisitState=VisitState())
    (has_visited_out(s, v) || Δ == 0) && return

    visited_out!(s, v)
    Δnout(v.meta, Δ)

    for vi in outputs(v)
        ins = inputs(vi)
        Δs = zeros(typeof(Δ), length(ins))
        Δs[findall(vx -> vx == v, ins)] .= Δ
        Δnin(vi, Δs...; s=s)
    end
 end
