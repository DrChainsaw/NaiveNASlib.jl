
"""
    AbstractConnectStrategy

Base type for strategies for how to (re)connect vertices when doing structural
mutation.
"""
abstract type AbstractConnectStrategy end
struct ConnectAll <: AbstractConnectStrategy end
struct ConnectNone <: AbstractConnectStrategy end

"""
    AbstractAlignSizeStrategy

Base type for strategies for how to align size (nin/nout) when doing structural mutation
"""
abstract type AbstractAlignSizeStrategy end
"""
    IncreaseSmaller

Try to align size by increasing in the direction (in/out) which has the smaller size.
Fallback to another strategy (default DecreaseBigger) if size change is not possible.
"""
struct IncreaseSmaller <: AbstractAlignSizeStrategy
    fallback
end
IncreaseSmaller() = IncreaseSmaller(DecreaseBigger(FailAlignSize()))

"""
    DecreaseBigger

Try to align size by decreasing in the direction (in/out) which has the bigger size.
Fallback to another strategy (default FailAlignSize) if size change is not possible.
"""
struct DecreaseBigger <: AbstractAlignSizeStrategy
    fallback
end
DecreaseBigger() = DecreaseBigger(FailAlignSize())

"""
    ChangeNinOfOutputs

Just sets nin of each output to the provided value.
"""
struct ChangeNinOfOutputs <: AbstractAlignSizeStrategy
    Δoutsize
end
"""
    FailAlignSize

Throws an error.
"""
struct FailAlignSize <: AbstractAlignSizeStrategy end

"""
    RemoveStrategy

Strategy for removal of a vertex.

Consists of an AbstractConnectStrategy for how to treat inputs and outputs of
the removed vertex and an AbstractAlignSizeStrategy for how to align sizes of
inputs and outputs.
"""
struct RemoveStrategy
    reconnect::AbstractConnectStrategy
    align::AbstractAlignSizeStrategy
end
RemoveStrategy() = RemoveStrategy(ConnectAll(), IncreaseSmaller())
RemoveStrategy(rs::AbstractConnectStrategy) = RemoveStrategy(rs, IncreaseSmaller())
RemoveStrategy(as::AbstractAlignSizeStrategy) = RemoveStrategy(ConnectAll(), as)

"""
    remove!(v::AbstractMutationVertex, strategy=RemoveStrategy())

Removes v from the graph by removing it from its inputs and outputs.

It is possible to supply a strategy for how to 1) reconnect the inputs and outputs
of v and 2) align the input and output sizes of the inputs and outputs of v.

Default strategy is to first set nin==nout for v and then connect all its inputs
to all its outputs.
"""
function remove!(v::AbstractMutationVertex, strategy=RemoveStrategy())
    prealignsizes(strategy.align, v)
    remove!(v, inputs, outputs, strategy.reconnect)
    remove!(v, outputs, inputs, strategy.reconnect)
    postalignsizes(strategy.align, v)
end

## Helper function to avoid code duplication. I don't expect this to be able to do
## anything useful unless f1=inputs and f2=outputs or vise versa.
function remove!(v::AbstractMutationVertex, f1::Function, f2::Function, s::AbstractConnectStrategy)
    for v1 in f1(v)
        v1_2 = f2(v1)
        inds = findall(vx -> vx == v, v1_2)
        deleteat!(v1_2, inds)
        # E.g connect f2 of v to f2 of v1, e.g connect the inputs to v as inputs to v1 if f2 == inputs
        connect!(v1, v1_2, inds, f2(v), s)
    end
end


"""
    connect!(v, to, inds, items, ::AbstractConnectStrategy)

Does connection of "items" to "to" at position "inds" depending on the given strategy.
"""
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
prealignsizes(::FailAlignSize, v) = error("Could not align sizes of $(v)!")

function postalignsizes(s::AbstractAlignSizeStrategy, v) end
postalignsizes(::FailAlignSize, v) = error("Could not align sizes of $(v)!")
