
"""
    base(v::AbstractVertex)

Return base vertex
"""
function base end

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
clone(v::OutputsVertex, ins::AbstractVertex...) = OutputsVertex(clone(base(v), ins...))

base(v::OutputsVertex) = v.base
(v::OutputsVertex)(x...) = base(v)(x...)

inputs(v::OutputsVertex) = inputs(base(v))
outputs(v::OutputsVertex) = v.outs

"""
    InputSizeVertex

Vertex with an (immutable) size. Intended use if for wrapping an InputVertex
in conjuntion with mutation
"""
struct InputSizeVertex <: AbstractVertex
    base::AbstractVertex
    size::Integer

    function InputSizeVertex(b::OutputsVertex, size::Integer)
        this = new(b, size)
        init!(b, this)
        return this
    end
end
InputSizeVertex(name, size::Integer) = InputSizeVertex(InputVertex(name), size)
InputSizeVertex(b::AbstractVertex, size::Integer) = InputSizeVertex(OutputsVertex(b), size)
clone(v::InputSizeVertex, ins::AbstractVertex...) = InputSizeVertex(clone(base(v), ins...), v.size)

base(v::InputSizeVertex)::AbstractVertex = v.base
(v::InputSizeVertex)(x...) = base(v)(x...)

inputs(v::InputSizeVertex) = inputs(base(v))
outputs(v::InputSizeVertex) = outputs(base(v))


abstract type MutationTrait end
clone(t::MutationTrait) = t

struct Immutable <: MutationTrait end
trait(v::AbstractVertex) = Immutable()

struct MutationVertex <: AbstractVertex
    base::AbstractVertex
    op::MutationOp
    trait::MutationTrait

    function MutationVertex(b::OutputsVertex, s::MutationOp, t::MutationTrait)
        this = new(b, s, t)
        init!(b, this)
        return this
    end
end
MutationVertex(b::AbstractVertex, s::MutationOp, t::MutationTrait) = MutationVertex(OutputsVertex(b), s, t)

AbsorbVertex(b::AbstractVertex, s::MutationState) = MutationVertex(b, s, SizeAbsorb())

StackingVertex(b::AbstractVertex) = StackingVertex(OutputsVertex(b))
StackingVertex(b::Union{OutputsVertex, MutationVertex}) = MutationVertex(b, IoSize(nout.(inputs(b)), sum(nout.(inputs(b)))), SizeStack())

InvariantVertex(b::AbstractVertex) = InvariantVertex(OutputsVertex(b), NoOp())
InvariantVertex(b::AbstractVertex, op::MutationOp) = MutationVertex(OutputsVertex(b), op, SizeInvariant())

clone(v::MutationVertex, ins::AbstractVertex...; opfun=cloneop, traitfun=clonetrait) = MutationVertex(clone(base(v), ins...), opfun(v), traitfun(v))
cloneop(v::MutationVertex) = clone(op(v))
clonetrait(v::MutationVertex) = clone(trait(v))

base(v::MutationVertex) = v.base
op(v::MutationVertex) = v.op
trait(v::MutationVertex) = v.trait
(v::MutationVertex)(x...) = base(v)(x...)

inputs(v::MutationVertex)  = inputs(base(v))
outputs(v::MutationVertex) = outputs(base(v))
