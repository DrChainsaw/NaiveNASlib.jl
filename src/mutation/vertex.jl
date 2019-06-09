
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
    outs::AbstractVector{AbstractVertex}
end
OutputsVertex(v::AbstractVertex) = OutputsVertex(v, AbstractVertex[])
init!(v::OutputsVertex, p::AbstractVertex) = foreach(in -> push!(outputs(in), p), inputs(v))
clone(v::OutputsVertex, ins::AbstractVertex...) = OutputsVertex(clone(base(v), ins...))

base(v::OutputsVertex) = v.base
(v::OutputsVertex)(x...) = base(v)(x...)

inputs(v::OutputsVertex) = inputs(base(v))
outputs(v::OutputsVertex) = v.outs

show_less(io::IO, v::OutputsVertex) = show_less(io, base(v))

function show(io::IO, v::OutputsVertex)
     show(io, base(v))
     print(io, ", outputs=")
     show(io, outputs(v))
 end

"""
    InputSizeVertex

Vertex with an (immutable) size.

Intended use is for wrapping an InputVertex in conjuntion with mutation
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

show_less(io::IO, v::InputSizeVertex) = show_less(io, base(v))



"""
    MutationTrait

Base type for traits relevant when mutating.
"""
abstract type MutationTrait end
# For convenience as 99% of all traits are immutable. Don't forget to implement for stateful traits or else there will be pain!
clone(t::MutationTrait) = t

"""
    Immutable

Trait for vertices which are immutable. Typically inputs and outputs as those are fixed to the surroundings (e.g a data set).
"""
struct Immutable <: MutationTrait end
trait(v::AbstractVertex) = Immutable()

abstract type DecoratingTrait <: MutationTrait end

struct NamedTrait <: DecoratingTrait
    base::MutationTrait
    name
end
base(t::NamedTrait) = t.base

"""
    MutationVertex

Vertex which may be subject to mutation.

Scope is mutations which affect the input and output sizes as such changes needs to be propagated to the neighbouring vertices.

The member op describes the type of mutation, e.g if individual inputs/outputs are to be pruned vs just changing the size without selecting any particular inputs/outputs.

The member trait describes the nature of the vertex itself, for example if size changes
are absorbed (e.g changing an nin x nout matrix to an nin - Δ x nout matrix) or if they
propagate to neighbouring vertices (and if so, how).
"""
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

AbsorbVertex(b::AbstractVertex, s::MutationState, f::Function=identity) = MutationVertex(b, s, f(SizeAbsorb()))

StackingVertex(b::AbstractVertex, f::Function=identity) = StackingVertex(OutputsVertex(b), f)
StackingVertex(b::Union{OutputsVertex, MutationVertex}, f::Function=identity) = MutationVertex(b, IoSize(nout.(inputs(b)), sum(nout.(inputs(b)))), f(SizeStack()))

InvariantVertex(b::AbstractVertex, f::Function=identity) = InvariantVertex(b, NoOp(), f)
InvariantVertex(b::AbstractVertex, op::MutationOp, f::Function=identity) = MutationVertex(OutputsVertex(b), op, f(SizeInvariant()))

clone(v::MutationVertex, ins::AbstractVertex...; opfun=cloneop, traitfun=clonetrait) = MutationVertex(clone(base(v), ins...), opfun(v), traitfun(v))
cloneop(v::MutationVertex) = clone(op(v))
clonetrait(v::MutationVertex) = clone(trait(v))

base(v::MutationVertex) = v.base
op(v::MutationVertex) = v.op
trait(v::MutationVertex) = v.trait
(v::MutationVertex)(x...) = base(v)(x...)

inputs(v::MutationVertex)  = inputs(base(v))
outputs(v::MutationVertex) = outputs(base(v))

show_less(io::IO, v::MutationVertex) = show_less(io, trait(v), v)
show_less(io::IO, ::MutationTrait, v::MutationVertex) = summary(io, v)
show_less(io::IO, t::NamedTrait, v::MutationVertex) = print(io, t.name)
show_less(io::IO, t::DecoratingTrait, v::MutationVertex) = show_less(io, base(t), v)
