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
    prealignsizes(strategy.align, v, vx -> vx == v)
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

# I don't like if-statements anymore, ok?
minΔninfactor_if(remove, v) = minΔninfactor_if(Val(remove), v)
minΔninfactor_if(::Val{true}, v) = lcmsafe(minΔnoutfactor_only_for.(inputs(v)))
minΔninfactor_if(::Val{false}, v) = minΔninfactor(v)

minΔnoutfactor_if(remove, v) = minΔnoutfactor_if(Val(remove), v)
minΔnoutfactor_if(::Val{true}, v) = lcmsafe(minΔninfactor_only_for.(outputs(v)))
minΔnoutfactor_if(::Val{false}, v) = minΔnoutfactor(v)

prealignsizes(s::AbstractAlignSizeStrategy, v, will_rm::Function) = prealignsizes(s, v, v, will_rm)
function prealignsizes(s::AbstractAlignSizeStrategy, vin, vout, will_rm) end
function prealignsizes(s::Union{IncreaseSmaller, DecreaseBigger}, vin, vout, will_rm)
    Δinsize = nout(vin) - tot_nin(vout)
    Δoutsize = -Δinsize

    can_change(Δ, factor::Integer) = Δ % factor == 0
    can_change(Δ, ::Missing) = false

    insize_can_change = all( can_change.(Δinsize, minΔninfactor_if(will_rm(vout), vout)))
    insize_can_change && proceedwith(s, Δinsize) && return Δnin(vout, Δinsize)

    outsize_can_change = all( can_change.(Δoutsize, minΔnoutfactor_if(will_rm(vin), vin)))
    outsize_can_change && proceedwith(s, Δoutsize)  && return Δnout(vin, Δoutsize)
    prealignsizes(s.fallback, vin, vout, will_rm)
end
proceedwith(::DecreaseBigger, Δ::Integer) = Δ <= 0
proceedwith(::IncreaseSmaller, Δ::Integer) = Δ >= 0

tot_nin(v) = tot_nin(trait(v), v)
tot_nin(t::DecoratingTrait, v) = tot_nin(base(t), v)
tot_nin(::MutationTrait, v) = nin(v)[]
tot_nin(::SizeInvariant, v) = unique(nin(v))[]
tot_nin(::SizeTransparent, v) = sum(nin(v))


prealignsizes(s::ChangeNinOfOutputs, vin, vout, will_rm) = Δnin.(outputs(vin), s.Δoutsize...)
prealignsizes(::FailAlignSize, vin, vout, will_rm) = error("Could not align sizes of $(v)!")

function prealignsizes(s::AlignSizeBoth, vin, vout, will_rm)

    Δninfactor = minΔninfactor_if(will_rm(vout), vout)
    Δnoutfactor = minΔnoutfactor_if(will_rm(vin), vin)

    ismissing(Δninfactor) || ismissing(Δnoutfactor) && return prealignsizes(s.fallback, vin, vout, will_rm)

    sizes = [nout(vin), tot_nin(vout)]
    accept(Δs) = all(-Δs .< sizes) || any(Δs .> 0.2 .* sizes)

    Δs = alignfor(nout(vin) , Δnoutfactor, [tot_nin(vout)], [Δninfactor], accept)

    # One last check if size change is possible
    -Δs[1] > nout(vin) && -Δs[2] > tot_nin(vout) &&  return prealignsizes(s.fallback, vin, vout, will_rm)

    # Ok, lets make the change
    s = VisitState{Int}() # Just in case we happen to be inside a transparent vertex
    Δnin(vout, Δs[2], s=s)
    Δnout(vin, Δs[1], s=s)
end

postalignsizes(s::AbstractAlignSizeStrategy, v) = postalignsizes(s, v, v)
function postalignsizes(s::AbstractAlignSizeStrategy, vin, vout) end
postalignsizes(::FailAlignSize, vin, vout) = error("Could not align sizes of $(vin) and $(vout)!")

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
    Δs = alignfor(nout(vin) , minΔnoutfactor_only_for(vin), nins, Δninfactors)

    ismissing(Δs) && postalignsizes(s.fallback, vin, vout)

    s = VisitState{Int}()
    visited_in!.(s, outputs(vout))
    Δnout(vin, Δs[1], s=s)

    for (i, voo) in enumerate(outputs(vout))
        # Dont touch the parts which already have the correct size
        s = VisitState{Int}()
        visited_out!(s, vout)
        Δvec = Vector{Maybe{Int}}(missing, length(inputs(voo)))
        Δvec[inputs(voo) .== vout] .= Δs[i+1]

        Δnin(voo, Δvec..., s=s)
    end

end

function all_positive(x, bound=200)
    minval,maxval = extrema(x)
    return minval >= 0 || maxval > bound
end

"""
    alignfor(nout, Δnoutfactor, nins, Δninfactors, accept = all_positive)

Returns `Δ` so that `vcat(nout, nins) .+ Δ |> unique |> length == 1` and so that `all(Δ .% vcat(Δnoutfactor, Δninfactors) .== 0)`.

In other words, return the `Δ` which makes `nout` equal to all `nins` while still being evenly divisible by the `Δfactors`.

Solves the following system of linear diophantine equations:
```text
nout + Δnoutfactor*x = nins + Δninfactors[0] * y0
nout + Δnoutfactor*x = nins + Δninfactors[1] * y1
...
```
where `Δ = [x, y0, y1, ...]`

# Examples

```
julia> Δ = NaiveNASlib.alignfor(2,2,[3], [3])
2-element Array{Int64,1}:
 4
 3

julia> [2, 3] .+ Δ
2-element Array{Int64,1}:
 6
 6

 julia> Δ .% [2, 3]
 2-element Array{Int64,1}:
  0
  0

julia> Δ = NaiveNASlib.alignfor(2,2,[3, 7], [11, 13])
3-element Array{Int64,1}:
 122
 121
 117

julia> [2, 3, 7] .+ Δ
3-element Array{Int64,1}:
 124
 124
 124

julia> Δ .% [2, 11, 13]
3-element Array{Int64,1}:
 0
 0
 0
```

"""
function alignfor(nout, Δnoutfactor, nins, Δninfactors, accept = all_positive)

    # System is
    # A:
    # Δnoutfactor -Δninfactors[0] 0               0 ...
    # Δnoutfactor  0              -Δninfactors[1] 0 ...
    # ...
    # B:
    # nins .- nout
    n = length(nins)
    A = hcat(repeat([Δnoutfactor], n), -Diagonal(Δninfactors))
    C = nins .- nout

    res = solve_lin_dio_eq(A,C)
    ismissing(res) && return res

    V, T, dof = res
    # So here's the deal:
    # [x, y0, y1, ...] =  V[:,1:end-dof] * T + V[:,end-dof+1:end] * K
    # K is the degrees of freedom due to overdetermined system
    # In fact, any K ∈ Z will yield a solution.
    # Which one do I want?

    # Answer: The one which makes the smallest change
    # As I'm really sick of this problem right now, lets
    # make a heuristic which should get near enough:
    # Solve the above system for [x, y0,..] = 0 without
    # an integer constraint, then round to nearest integer
    # and increase K until values are accepted. Note that
    # that last part probably only works for dof = 1, but
    # it always is for this system (as Δfactors are > 0) (or?)
    VT = V[:, 1:end-dof] * T
    VK = V[:,end-dof+1:end]
    K = round.(Int, -VK \ VT)

    f = k -> (VT + VK * k) .* vcat(Δnoutfactor, Δninfactors)

    # Search for an accepted solution
    Δs = f(K)
    while !accept(Δs)
        K .+= 1
        Δs = f(K)
    end
    return Δs
end


"""
    solve_lin_dio_eq(A,C)

Return `V`, `T`, `dof` so that `X` = `V` * `vcat(T,K)` and `AX` = `C` where `A`, `X` and `C` are matrices of integers where `K` is a vector of arbitrary integers of length `dof` or `missing` if no solution is possible.

Note that `dof` might be 0, meaning no freedom for you.

# Examples
```julia
julia> A = [1 2 3; 3 2 1; 2 4 6] # Note: A[1,:] == 2*A[3,:]
3×3 Array{Int64,2}:
 1  2  3
 3  2  1
 2  4  6

 julia> NaiveNASlib.solve_lin_dio_eq(A, [1,2,3]) # Note: B[1] != 2*[B3]
missing

julia> V, T, dof = NaiveNASlib.solve_lin_dio_eq(A, [1,3,2]) # Now it is!
([1 -5 1; 0 1 -2; 0 0 1], [3, 1], 1)

julia> A * V * vcat(T, ones(Int, dof))
3-element Array{Int64,1}:
 1
 3
 2

 julia> A * V * vcat(T, 666ones(Int, dof))
3-element Array{Int64,1}:
 1
 3
 2

```
"""
function solve_lin_dio_eq(A,C)

    B,U,V = snf_with_transform(matrix(ZZ, A))     # B = UAV

    # AbstractAlgebra works with BigInt...
    D = Int.(U.entries) * C

    BD = LinearAlgebra.diag(Int.(B.entries))
    k = findall(BD .!= 0)
    n = length(k)+1 : length(D)

    # Check that solution exists which is only if
    # 1) B[i.i] divides D[i] for i <= k
    # 2) D[i] = 0 for i > k
    all(D[k] .% BD[k] .== 0) || return missing
    all(D[n] .== 0) || return missing

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

    prealignsizes(strategy, from, to, v -> false)

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
