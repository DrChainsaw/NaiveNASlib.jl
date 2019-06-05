
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

cloneop(v::AbstractMutationVertex) = clone(op(v))


abstract type AbstractConnectStrategy end
struct ConnectAll <: AbstractConnectStrategy end
struct ConnectNone <: AbstractConnectStrategy end

abstract type AbstractAlignSizeStrategy end
struct IncreaseSmaller <: AbstractAlignSizeStrategy
    fallback
end
IncreaseSmaller() = IncreaseSmaller(DecreaseBigger(Fail()))
struct DecreaseBigger <: AbstractAlignSizeStrategy
    fallback
end
DecreaseBigger() = DecreaseBigger(Fail())
struct ChangeNinOfOutputs <: AbstractAlignSizeStrategy
    Δoutsize
end
struct Fail <: AbstractAlignSizeStrategy end

struct RemoveStrategy
    reconnect::AbstractConnectStrategy
    align::AbstractAlignSizeStrategy
end
RemoveStrategy() = RemoveStrategy(ConnectAll(), IncreaseSmaller())
RemoveStrategy(rs::AbstractConnectStrategy) = RemoveStrategy(rs, IncreaseSmaller())
RemoveStrategy(as::AbstractAlignSizeStrategy) = RemoveStrategy(ConnectAll(), as)

function remove!(v::AbstractMutationVertex, strategy=RemoveStrategy())
    prealignsizes(strategy.align, v)
    remove!(v, inputs, outputs, strategy.reconnect)
    remove!(v, outputs, inputs, strategy.reconnect)
    postalignsizes(strategy.align, v)
end

function remove!(v::AbstractMutationVertex, f1::Function, f2::Function, s::AbstractConnectStrategy)
    for v1 in f1(v)
        v1_2 = f2(v1)
        inds = findall(vx -> vx == v, v1_2)
        deleteat!(v1_2, inds)

        # E.g connect f2 of v to f2 of v1, e.g connect the inputs to v as inputs to v1 if f2 == inputs
        connect!(v1, v1_2, inds, f2(v), s)
    end
end

#Not sure broadcasting insert! like this is supposed to work, but it does...
connect!(v, to, inds, items, ::ConnectAll) = foreach(ind -> insert!.([to], [ind], items), inds)
function connect!(v, to, inds, items, ::ConnectNone) end

function prealignsizes(s::AbstractAlignSizeStrategy, v) end
function prealignsizes(s::Union{IncreaseSmaller, DecreaseBigger}, v)
    Δinsize = nout(v) - sum(nin(v))
    Δoutsize = -Δinsize

    insize_can_change = all(isa.(findabsorbing(Transparent(), v, inputs), AbstractMutationVertex))
    outsize_can_change = all(isa.(findabsorbing(Transparent(), v, outputs), AbstractMutationVertex))

    insize_can_change && proceedwith(s, Δinsize) && return Δnin(v, Δinsize)
    outsize_can_change && proceedwith(s, Δoutsize)  && return Δnout(v, Δoutsize)
    prealignsizes(s.fallback, v)
end
proceedwith(::DecreaseBigger, Δ::Integer) = Δ <= 0
proceedwith(::IncreaseSmaller, Δ::Integer) = Δ >= 0

prealignsizes(s::ChangeNinOfOutputs, v) = Δnin.(outputs(v), s.Δoutsize)
prealignsizes(::Fail, v) = error("Could not align sizes of $(v)!")

function postalignsizes(s::AbstractAlignSizeStrategy, v) end
postalignsizes(::Fail, v) = error("Could not align sizes of $(v)!")

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


"""
    AbsorbVertex

Vertex which absorbs changes in nout or nin. An example of this is a vertex
which multiplies its input with an nin x nout matrix.
"""
struct AbsorbVertex <: AbstractMutationVertex
    base::AbstractVertex
    state::MutationState

    function AbsorbVertex(b::OutputsVertex, state::MutationState)
        this = new(b, state)
        init!(b, this)
        return this
    end
end
AbsorbVertex(b::AbstractVertex, state::MutationState) = AbsorbVertex(OutputsVertex(b), state)

clone(v::AbsorbVertex, ins::AbstractVertex...; opfun=cloneop) = AbsorbVertex(clone(base(v), ins...), opfun(v))
op(v::AbsorbVertex) = v.state

base(v::AbsorbVertex)::AbstractVertex = v.base
(v::AbsorbVertex)(x...) = base(v)(x...)


"""
    StackingVertex

Vertex which is transparent w.r.t mutation.

Size of output is sum of sizes of inputs. Examples of computations are scalar operations
(e.g add x to every element) and concatenation.
"""
struct StackingVertex <: AbstractMutationVertex
    base::AbstractVertex
    state::MutationState

    function StackingVertex(b::OutputsVertex, op::MutationState)
        this = new(b, op)
        init!(b, this)
        return this
    end
end
StackingVertex(b::AbstractVertex) = StackingVertex(OutputsVertex(b))
StackingVertex(b::Union{OutputsVertex, AbstractMutationVertex}) = StackingVertex(b, IoSize(nout.(inputs(b)), sum(nout.(inputs(b)))))

clone(v::StackingVertex, ins::AbstractVertex...; opfun=cloneop) = StackingVertex(clone(base(v), ins...), opfun(v))
op(v::StackingVertex) = v.state

base(v::StackingVertex)::AbstractVertex = v.base
(v::StackingVertex)(x...) = base(v)(x...)


"""
    InvariantVertex

Vertex which is size invariant in the sense that all inputs and outputs have the same size.

Examples of computations are scalar and element wise operations.
"""
struct InvariantVertex <: AbstractMutationVertex
    base::AbstractVertex
    op::MutationOp

    function InvariantVertex(b::OutputsVertex, op::MutationOp)
        this = new(b, op)
        init!(b, this)
        return this
    end
end
InvariantVertex(b::AbstractVertex) = InvariantVertex(OutputsVertex(b), NoOp())
InvariantVertex(b::AbstractVertex, op::MutationOp) = InvariantVertex(OutputsVertex(b), op)

clone(v::InvariantVertex, ins::AbstractVertex...; opfun=cloneop) = InvariantVertex(clone(base(v), ins...), opfun(v))
op(v::InvariantVertex) = v.op

base(v::InvariantVertex)::AbstractVertex = v.base
(v::InvariantVertex)(x...) = base(v)(x...)
