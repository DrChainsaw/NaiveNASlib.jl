
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
    AbsorbVertex

Vertex which absorbs changes in nout or nin. An example of this is a vertex
which multiplies its input with an nin x nout matrix.
"""
struct AbsorbVertex <: AbstractMutationVertex
    base::AbstractVertex
    meta::VertexMeta
end
base(v::AbsorbVertex)::AbstractVertex = v.base

Δnin(v::AbsorbVertex, Δ::Integer...) = Δnin(v.meta, Δ...)
Δnout(v::AbsorbVertex, Δ::Integer) = Δnout(v.meta, Δ)
