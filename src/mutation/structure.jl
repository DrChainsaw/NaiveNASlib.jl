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

Base type for strategies for how to align size (nin/nout) when doing structural mutation.

Note that all strategies are not guaranteed to work in all cases.

Default strategies should however be selected based on case so that things always work out.
"""
abstract type AbstractAlignSizeStrategy end

"""
    NoSizeChange

Don't do any size change.
"""
struct NoSizeChange <: AbstractAlignSizeStrategy end

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
    AdjustOutputsToCurrentSize

Adjust nin of all outputs to the current output size while preventing changes from happening to inputs.

This is a post-align strategy, i.e it will be applied after a structural change has been made.
"""
struct AdjustOutputsToCurrentSize <: AbstractAlignSizeStrategy
    fallback
end
AdjustOutputsToCurrentSize() = AdjustOutputsToCurrentSize(FailAlignSize())

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
    for f1_v in f1(v)
        f1_f2_v = f2(f1_v)
        inds = findall(vx -> vx == v, f1_f2_v)
        deleteat!(f1_f2_v, inds)
        # Connect f2 of v to f2 of v1, e.g connect the inputs to v as inputs to v1 if f2 == inputs
        connect!(f1_v, f1_f2_v, inds, f2(v), s)
    end
end


"""
    connect!(v, to, inds, items, ::AbstractConnectStrategy)

Does connection of "items" to "to" at position "inds" depending on the given strategy.
"""
#Not sure broadcasting insert! like this is supposed to work, but it does...
connect!(v, to, inds, items, ::ConnectAll) = insert!.([to], inds, items)
function connect!(v, to, inds, items, ::ConnectNone)
    # Stupid generic function forces me to check this...
    # ... and no, it is not easy to understand what is going on here...
    if to == inputs(v) && !isempty(inds)
        rem_input!.(op.(v), inds...)
    end
end

prealignsizes(s::AbstractAlignSizeStrategy, v) = prealignsizes(s, v, v)
function prealignsizes(s::AbstractAlignSizeStrategy, vin, vout) end
function prealignsizes(s::Union{IncreaseSmaller, DecreaseBigger}, vin, vout)
    Δinsize = nout(vin) - tot_nin(vout)
    Δoutsize = -Δinsize

    can_change(Δ, factor::Integer) = Δ % factor == 0
    can_change(Δ, ::Missing) = false

    insize_can_change = all( can_change.(Δinsize, minΔnoutfactor_only_for.(inputs(vout))))
    insize_can_change && proceedwith(s, Δinsize) && return Δnin(vout, Δinsize)

    outsize_can_change = all( can_change.(Δoutsize, minΔninfactor_only_for.(outputs(vin))))
    outsize_can_change && proceedwith(s, Δoutsize)  && return Δnout(vin, Δoutsize)
    prealignsizes(s.fallback, vin, vout)
end
proceedwith(::DecreaseBigger, Δ::Integer) = Δ <= 0
proceedwith(::IncreaseSmaller, Δ::Integer) = Δ >= 0

tot_nin(v) = tot_nin(trait(v), v)
tot_nin(t::DecoratingTrait, v) = tot_nin(base(t), v)
tot_nin(::MutationTrait, v) = nin(v)[]
tot_nin(::SizeInvariant, v) = unique(nin(v))[]
tot_nin(::SizeTransparent, v) = sum(nin(v))


prealignsizes(s::ChangeNinOfOutputs, vin, vout) = Δnin.(outputs(vin), s.Δoutsize...)
prealignsizes(::FailAlignSize, vin, vout) = error("Could not align sizes of $(v)!")

function prealignsizes(s::AlignSizeBoth, vin, vout)

    Δninfactor = lcmsafe(minΔnoutfactor_only_for.(inputs(vout)))
    Δnoutfactor = lcmsafe(minΔninfactor_only_for.(outputs(vin)))
    ismissing(Δninfactor) || ismissing(Δnoutfactor) && return prealignsizes(s.fallback, vin, vout)

    # Ok, for this to work out, we need sum(nin(v)) + Δnin == nout(v) + Δnout where
    # Δnin = Δninfactor * x and Δnout = Δnoutfactor * y (therefore we we need x, y ∈ Z).
    # This can be rewritten as a linear diophantine equation a*x + b*y = c where a = Δninfactor, b = -Δnoutfactor and c = nout(v) - sum(nin(v))
    # Thank you julia for having a built in solver for it (gcdx)

    # Step 1: Rename to manageble variable names
    a = Δninfactor
    b = -Δnoutfactor
    c = nout(vin) - tot_nin(vout)

    # Step 2: Calculate gcd and Bézout coefficients
    (d, p, q) = gcdx(a,b)

    # Step 3: Check that the equation has a solution
    c % d == 0 || return prealignsizes(s.fallback, vin, vout)

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
    #   min wrt k: (a(x + bk))^2 + (b(y - ak))^2
    # Just round the result, it should be close enough
    k = round(Int, 2a*b*(b*y - a*x) / (4a^2*b^2))

    # Step 5: Fine tune if needed
    while -Δnin_f(k) > tot_nin(vout);  k += sign(b) end
    while -Δnout_f(k) > nout(vin); k -= sign(a) end

    # Step 6: One last check if size change is possible
    -Δnin_f(k) > sum(nin(vout)) && -Δnout_f(k) > nout(vin) && return prealignsizes(s.fallback, vin, vout)

    # Step 7: Make the change
     s = VisitState{Int}() # Just in case we happen to be inside a transparent vertex
    Δnin(vout, Δnin_f(k), s=s)
    Δnout(vin, Δnout_f(k), s=s)
end

postalignsizes(s::AbstractAlignSizeStrategy, v) = postalignsizes(s, v, v)
function postalignsizes(s::AbstractAlignSizeStrategy, vin, vout) end
postalignsizes(::FailAlignSize, v) = error("Could not align sizes of $(v)!")

postalignsizes(s::AdjustOutputsToCurrentSize, vin, vout) = postalignsizes(s, vin, vout, trait(vout))
postalignsizes(s::AdjustOutputsToCurrentSize, vin, vout, t::DecoratingTrait) = postalignsizes(s, vin, vout, base(t))

function postalignsizes(s::AdjustOutputsToCurrentSize, vin, vout, ::SizeStack)

    # At this point we expect to have nout(vout) - nout(vin) = nin of all output edges of vout and we need to fix this so nout(vout) == nin while nin(vout) = nout of all its inputs.

    # Therefore the equations we want to solve are nout(vin) + Δnoutfactor * x + nin(voo[i]) = nin(voo[i]) +  Δninfactor[i] * y[i] for where voo[i] is output #i of vout. As nin(voo[i]) is eliminated this leaves us with the zeros below.
    nins = zeros(Int, length(outputs(vout)))
    Δninfactors = minΔninfactor_only_for.(outputs(vout))

    # This probably does not have to be this complicated.
    # All nins are the same and then one could just do a single linear
    # diophantine equation with lcm of all Δninfactors
    X = alignfor(nout(vin) , minΔnoutfactor_only_for(vin), nins, Δninfactors)

    ismissing(X) && postalignsizes(s.fallback, vin, vout)

    s = VisitState{Int}()
    visited_in!.(s, outputs(vout))
    Δnout(vin, X[1], s=s)

    for (i, voo) in enumerate(outputs(vout))
        # Dont touch the parts which already have the correct size
        s = VisitState{Int}()
        visited_out!(s, vout)
        Δvec = Vector{Maybe{Int}}(missing, length(inputs(voo)))
        Δvec[inputs(voo) .== vout] .= X[i+1]

        Δnin(voo, Δvec..., s=s)
    end

end

function alignfor(nout, Δnoutfactor, nins, Δninfactors)
    # System of linear diophantine equations.. fml!
    # nout + Δnoutfactor*x = nins + Δninfactors[0] * y0
    # nout + Δnoutfactor*x = nins + Δninfactors[1] * y1
    # ...
    # Thus we have:
    # A:
    # Δnoutfactor -Δninfactors[0] 0               0 ...
    # Δnoutfactor  0              -Δninfactors[1] 0 ...
    # ...
    # B:
    # nout - nins[0]
    n = length(nins)
    A = hcat(repeat([Δnoutfactor], n), -Diagonal(Δninfactors))
    B = nins .- nout

    res = solve_lin_dio_eq(A,B)
    ismissing(res) && return res

    V, T, dof = res
    # So here's the deal:
    # [x, y0, y1, ...] =  V[:,1:end-dof] * T + V[:,end-dof+1:end] * K
    # K is the degrees of freedom due to overdetermined system
    # In fact, any K ∈ Z will yield a solution.
    # Which one do I want?

    # Answer: The one which makes the smallest but positive change
    # As I'm really sick of this problem right now, lets
    # make a heuristic which should get me near enough:
    # Solve the above system for [x, y0,..] = 0 without
    # an integer constraint, then round to nearest integer
    # and increase K until values are positive. Note that
    # that last part probably only works for dof = 1, but
    # it always is for this system (as Δfactors are > 0) (or?)
    VT = V[:, 1:end-dof] * T
    VK = V[:,end-dof+1:end]
    K = round.(Int, -VK \ VT)

    f = k -> (VT + VK * k)

    # Search for non-negatives within bounds
    X = f(K)
    within_bounds(a,b) = a < 0 && b < 100
    while within_bounds(extrema(X)...)
        K .+= 1
        X = f(K)
    end
    return X .* vcat(Δnoutfactor, Δninfactors)
end


# Return V, T, dof so that X = V * vcat(T, K) and AX = C where A, X and C are matrices of integers where K is a vector of arbitrary integers of length dof (note that dof might be 0 though meaning no freedom for you)
function solve_lin_dio_eq(A,C)
    B,U,V = snf_with_transform(matrix(ZZ, A))     # B = UAV

    # AbstractAlgebra works with BigInt...
    D = Int.(U.entries) * C

    # Check that solution exists which is only if
    # 1) B[i.i] divides D[i] for i <= k
    # 2) D[i] = 0 for i > k
    BD = LinearAlgebra.diag(Int.(B.entries))
    k = findall(BD .!= 0)
    n = length(k)+1 : min(size(A)...)

    all(D[k] .% BD[k] .== 0) ||  all(D[n] .== 0) || return missing

    # We have a solution!
    return Int.(V.entries), div.(D[k], BD[k]), size(A,2) - length(k)
end

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

function create_edge!(from::AbstractVertex, to::AbstractVertex, pos = length(inputs(to))+1, strategy = default_create_edge_strat(to))

    prealignsizes(strategy, from, to)

    push!(outputs(from), to) # Order should never matter for outputs
    insert!(inputs(to), pos, from)

    add_input!(op(to), pos, nout(from))
    add_output!(op(to), trait(to), nout(from))

    postalignsizes(strategy, from, to)
end

default_create_edge_strat(v::AbstractVertex) = default_create_edge_strat(trait(v),v)
default_create_edge_strat(t::DecoratingTrait,v) = default_create_edge_strat(base(t),v)
default_create_edge_strat(::SizeStack,v) = AdjustOutputsToCurrentSize()
default_create_edge_strat(::SizeInvariant,v) = IncreaseSmaller()

function add_input!(::MutationOp, pos, size) end
add_input!(s::IoSize, pos, size) = insert!(s.nin, pos, size)
add_input!(s::IoIndices, pos, size) = insert!(s.in, pos, collect(1:size+1))

function add_output!(::MutationOp, t::MutationTrait, size) end
add_output!(s::MutationOp, t::DecoratingTrait, size) = add_output!(s, base(t), size)
add_output!(s::IoSize, ::SizeStack, size) = Δnout(s, size)
add_output!(s::IoIndices, ::SizeStack, size) = Δnout(s, vcat(s.out, (length(s.out):length(s.out)+size)))


function rem_input!(::MutationOp, pos...) end
rem_input!(s::Union{IoSize, IoIndices}, pos...) = deleteat!(s.nin, collect(pos))
