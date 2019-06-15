# Overengineered set of strategy types and structs? Not gonna argue with that, but I do this for fun and sometimes I have a wierd idea of what fun is.
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
    ChangeNinOfOutputs

Just sets nin of each output to the provided value. Sometimes you just know the answer...
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
    AlignSizeBoth

Align sizes by changing both input and output considering any Δfactors.
Fallback to another strategy (default FailAlignSize) if size change is not possible.
"""
struct AlignSizeBoth <: AbstractAlignSizeStrategy
    fallback
end
AlignSizeBoth() = AlignSizeBoth(FailAlignSize())

"""
    DecreaseBigger

Try to align size by decreasing in the direction (in/out) which has the bigger size.
Fallback to another strategy (default AlignSizeBoth) if size change is not possible.
"""
struct DecreaseBigger <: AbstractAlignSizeStrategy
    fallback
end
DecreaseBigger() = DecreaseBigger(AlignSizeBoth())

"""
    IncreaseSmaller

Try to align size by increasing in the direction (in/out) which has the smaller size.
Fallback to another strategy (default DecreaseBigger) if size change is not possible.
"""
struct IncreaseSmaller <: AbstractAlignSizeStrategy
    fallback
end
IncreaseSmaller() = IncreaseSmaller(DecreaseBigger())

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
    remove!(v::MutationVertex, strategy=RemoveStrategy())

Removes v from the graph by removing it from its inputs and outputs.

It is possible to supply a strategy for how to 1) reconnect the inputs and outputs
of v and 2) align the input and output sizes of the inputs and outputs of v.

Default strategy is to first set nin==nout for v and then connect all its inputs
to all its outputs.
"""
function remove!(v::MutationVertex, strategy=RemoveStrategy())
    prealignsizes(strategy.align, v)
    remove!(v, inputs, outputs, strategy.reconnect)
    remove!(v, outputs, inputs, strategy.reconnect)
    postalignsizes(strategy.align, v)
end

## Helper function to avoid code duplication. I don't expect this to be able to do
## anything useful unless f1=inputs and f2=outputs or vise versa.
function remove!(v::MutationVertex, f1::Function, f2::Function, s::AbstractConnectStrategy)
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

    can_change(Δ, factor::Integer) = Δ % factor == 0
    can_change(Δ, ::Missing) = false

    insize_can_change = all( can_change.(Δinsize, minΔnoutfactor_only_for.(inputs(v))))
    insize_can_change && proceedwith(s, Δinsize) && return Δnin(v, Δinsize)

    outsize_can_change = all( can_change.(Δoutsize, minΔninfactor_only_for.(outputs(v))))
    outsize_can_change && proceedwith(s, Δoutsize)  && return Δnout(v, Δoutsize)
    prealignsizes(s.fallback, v)
end
proceedwith(::DecreaseBigger, Δ::Integer) = Δ <= 0
proceedwith(::IncreaseSmaller, Δ::Integer) = Δ >= 0

prealignsizes(s::ChangeNinOfOutputs, v) = Δnin.(outputs(v), s.Δoutsize...)
prealignsizes(::FailAlignSize, v) = error("Could not align sizes of $(v)!")

function prealignsizes(s::AlignSizeBoth, v)

    Δninfactor = lcmsafe(minΔnoutfactor_only_for.(inputs(v)))
    Δnoutfactor = lcmsafe(minΔninfactor_only_for.(outputs(v)))
    ismissing(Δninfactor) || ismissing(Δnoutfactor) && return prealignsizes(s.fallback, v)

    # Ok, for this to work out, we need sum(nin(v)) + Δnin == nout(v) + Δnout where
    # Δnin = Δninfactor * x and Δnout = Δnoutfactor * y (therefore we we need x, y ∈ Z).
    # This can be rewritten as a linear diophantine equation a*x + b*y = c where a = Δninfactor, b = -Δnoutfactor and c = nout(v) - sum(nin(v))
    # Thank you julia for having a built in solver for it (gcdx)

    # Step 1: Rename to manageble variable names
    a = Δninfactor
    b = -Δnoutfactor
    c = nout(v) - sum(nin(v))

    # Step 2: Calculate gcd and Bézout coefficients
    (d, p, q) = gcdx(a,b)

    # Step 3: Check that the equation has a solution
    c % d == 0 || return prealignsizes(s.fallback, v)

    # Step 4 get base values
    x = div(c, d) * p
    y = div(c, d) * q

    # Step 4: Try to find the smallest Δnin and Δnout
    # We now have the solutions:
    #   Δnin  =  a(x + bk)
    #   Δnout = -b(y - ak) (b = -Δnoutfactor, remember?)
    Δnin_f = k -> a*(x + b*k)
    Δnout_f = k -> -b*(y - a*k)

    # Lets minimize the sum of squares:
    #   min wrt k: a(x + bk)^2 + b(y - ak)^2
    # Just round the result, it should be close enough
    k = round(Int,-2*a*b*(x + y) / (2a*b^2 - 2a^2*b))

    # Step 5: Fine tune if needed
    while -Δnin_f(k) > sum(nin(v));  k += 1 end
    while -Δnout_f(k) > nout(v); k -= 1 end

    # Step 6: One last check if size change is possible
    -Δnin_f(k) > sum(nin(v)) && -Δnout_f(k) > nout(v) && return prealignsizes(s.fallback, v)

    # Step 7: Make the change
    s = VisitState{Int}() # Just in case we happen to be inside a transparent vertex
    Δnin(v, Δnin_f(k), s=s)
    Δnout(v, Δnout_f(k), s=s)
end

function postalignsizes(s::AbstractAlignSizeStrategy, v) end
postalignsizes(::FailAlignSize, v) = error("Could not align sizes of $(v)!")

"""
    insert!(vin::AbstractVertex, factory::Function, outselect::Function=identity)

Replace `vin` as input to all outputs of `vin` with vertex produced by `factory`

Example:

```text
Before:

    vin
   / | \\
  /  |  \\
 v₁ ... vₙ

After:

    vin
     |
   vnew₁
     ⋮
   vnewₖ
   / | \\
  /  |  \\
 v₁ ... vₙ
```
Note that the connection `vin` -> `vnew₁` as well as all connections `vnewₚ -> vnewₚ₊₁` is done by `factory` in the above example.

The function `outselect` can be used to select a subset of outputs to replace (default all).

"""
function Base.insert!(vin::AbstractVertex, factory::Function, outselect::Function=identity)

    prevouts = copy(outselect(outputs(vin)))
    deleteat!(outputs(vin), findall(vx -> vx in prevouts, outputs(vin)))

    vnew = factory(vin)
    for vout in prevouts
        inds = findall(vx -> vx == vin, inputs(vout))
        inputs(vout)[inds] .= vnew
        push!(outputs(vnew), vout)
    end
end

function create_edge!(from::AbstractVertex, to::AbstractVertex, pos = length(inputs(to))+1 )
    push!(outputs(from), to) # Order should never matter for outputs
    insert!(inputs(to), pos, from)
    add_input(op(to), pos)

    # Now adjust the size.
    Δnins = zeros(Int, length(inputs(to)))
    Δnins[pos] = nout(from)

    # Dont touch the inputs!
    s = VisitState{Int}()
    visited_out!.(s, inputs(to))
    Δnin(to, Δnins..., s=s)
end

function add_input(::MutationOp, pos, val=nothing) end
add_input(s::IoSize, pos, val = 0) = insert!(s.nin, pos, val)
add_input(s::IoIndices, pos, val = []) = insert!(s.in, pos, val)
