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
prealignsizes(::FailAlignSize, vin, vout, will_rm) = error("Could not align sizes of $(vin) and $(vout)!")

function prealignsizes(s::AlignSizeBoth, vin, vout, will_rm)

    Δninfactor = minΔninfactor_if(will_rm(vout), vout)
    Δnoutfactor = minΔnoutfactor_if(will_rm(vin), vin)

    ismissing(Δninfactor) || ismissing(Δnoutfactor) && return prealignsizes(s.fallback, vin, vout, will_rm)

    sizes = [nout(vin), tot_nin(vout)]
    select(f,k) = increase_until(Δs -> all(-Δs .< sizes) || any(Δs .> 0.2 .* sizes), f, k)

    Δs = alignfor(nout(vin) , Δnoutfactor, [tot_nin(vout)], [Δninfactor], select)

    # One last check if size change is possible
    ismissing(Δs) &&  return prealignsizes(s.fallback, vin, vout, will_rm)
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

    # The equations we want to solve are nout(vin) + Δnoutfactor * x + nin(voo[i]) = nin(voo[i]) +  Δninfactor[i] * y[i] for where voo[i] is output #i of vout. As nin(voo[i]) is eliminated this leaves us with the zeros below.
    Δninfactors = minΔninfactor_only_for.(outputs(vout))
    nins = zeros(Int, length(Δninfactors))

    # What this!? Well, we do want to change vin and vout only, but perhaps this is not possible for one reason or the other (e.g. vin is immutable).
    # Therefore we will try to change each input to vout, starting with vin, until we find one which succeeds.

    # Some implementations of unique give sorted output, but this one preserves the order.
    vins = unique(vcat(vin, inputs(vout)))

    function alignsize(cnt=1)
        # Base case: We have tried all inputs and no possible solution was found
        cnt > length(vins) && return missing

        Δnoutfactors =minΔnoutfactor_only_for.(vins[cnt])

        select(f,k) = increase_until(all_positive, f, k)
        Δs = alignfor(nout(vin), Δnoutfactors, nins, Δninfactors, select)

        return ismissing(Δs) ? alignsize(cnt+1) : (Δs, cnt, cnt)
    end

    # TODO: This function and the one above can probably be merged
    # Obstacles:
    #     increase_until only works when Δninfactors are missing and there is only one nout
    #     Search order? First try all nins and then try combinations of them?
    #       -Doesn't it do that already?
    function alignsize_all_inputs(start=1, stop=2)
        stop > length(vins) && return missing
        Δnoutfactors =minΔnoutfactor_only_for.(vins[start:stop])

        Δs = alignfor(nout(vin), Δnoutfactors, nins, Δninfactors, select_start)

        if ismissing(Δs)
            return alignsize_vin_only(start+1, max(1, stop-1))
        elseif any(-Δs[1:stop-start+1] .>= nout.(vins[start:stop]))
            return alignsize_vin_only(start, stop+1)
        end
        return Δs, start, stop
    end

    res = any(ismissing.(Δninfactors)) ? alignsize_all_inputs() : alignsize()
    ismissing(res) && return postalignsizes(s.fallback, vin, vout)

    Δs, start, stop = res

    for i in start:stop
        s = VisitState{Int}()
        visited_in!.(s, outputs(vout))
        Δnout(vins[i], Δs[i-start+1], s=s)
    end

    for (i, voo) in enumerate(outputs(vout))
        Δ = Δs[i+stop-start+1]
        Δ == 0 && continue
        # Dont touch the parts which already have the correct size
        s = VisitState{Int}()
        visited_out!(s, vout)
        Δvec = Vector{Maybe{Int}}(missing, length(inputs(voo)))
        Δvec[inputs(voo) .== vout] .= Δ

        Δnin(voo, Δvec..., s=s)
    end

end

function increase_until(cond, f, k_start)
    # This is pretty coarse and is far from guaranteed to make the solution better
    k = k_start
    last = -Inf
    while true
        Δs = f(k)
        cond(Δs) && return Δs
        sum(Δs) .<= sum(last) && return last

        last = Δs
        k .+= 1
    end
end

function all_positive(x, ubound=200)
    minval,maxval = extrema(x)
    return minval >= 0 || maxval > ubound
end

select_start(f, k) = f(k)


"""
    alignfor(nouts, Δnoutfactors, nins, Δninfactors, select = all_positive)

Returns `Δ` so that `vcat(nouts, nins) .+ Δ |> unique |> length == 1` and so that `all(Δ .% vcat(Δnoutfactors, Δninfactors) .== 0)`.

In other words, return the `Δ` which makes `nouts` equal to all `nins` while still being evenly divisible by the `Δfactors`.

Solves the following system of linear diophantine equations:
```text
nouts[0] + Δnoutfactors[0]*x0 + nouts[1] + Δnoutfactors[1]*x1 + ... = nins + Δninfactors[0] * y0
nouts[0] + Δnoutfactors[0]*x0 + nouts[1] + Δnoutfactors[1]*x1 + ... = nins + Δninfactors[1] * y1
...
```
where `Δ = [x0, x1, ..., y0, y1, ...]`

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
function alignfor(nouts::Vector{<:Integer}, Δnoutfactors::Vector{<:Integer}, nins::Vector{<:Integer}, Δninfactors::Vector{<:Integer}, select = select_start)

    # System is
    # A:
    # Δnoutfactors -Δninfactors[0] 0               0 ...
    # Δnoutfactors  0              -Δninfactors[1] 0 ...
    # ...
    # B:
    # nins .- nouts
    n = length(nins)
    A = hcat(repeat(Δnoutfactors', n), -Diagonal(Δninfactors))

    C = nins .- sum(nouts)

    res = solve_lin_dio_eq(A,C)
    ismissing(res) && return missing

    V, T, dof = res

    dof == 0 && return V * T
    # So here's the deal:
    # [x, y0, y1, ...] =  V[:,1:end-dof] * T + V[:,end-dof+1:end] * K
    # K is the degrees of freedom due to overdetermined system
    # In fact, any K ∈ Z will yield a solution.
    # Which one do I want?

    # Answer: The one which makes the smallest change
    # As I'm really sick of this problem right now, lets
    # make a heuristic which should get near enough:
    # Solve the above system for [x, y0,..] = 0 without
    # an integer constraint, then round to nearest integer.
    VT = V[:, 1:end-dof] * T
    VK = V[:,end-dof+1:end]
    K = round.(Int, -VK \ VT)

    f = k -> (VT + VK * k) .* vcat(Δnoutfactors, Δninfactors)

    return select(f, K)
end

alignfor(nouts::Integer, Δnoutfactors::Integer, nins, Δninfactors, select = (f, k) -> increase_until(all_positive, f, k)) = alignfor([nouts], [Δnoutfactors], nins,  Δninfactors, select)
alignfor(nouts::Integer, Δnoutfactors::Vector, nins, Δninfactors, select = select_start) = alignfor([nouts], Δnoutfactors, nins,  Δninfactors, select)

alignfor(nouts, ::Missing, nins, Δninfactors, select = select_start) = alignfor(nouts, 0, nins, Δninfactors, select)


function alignfor(nouts::Vector{<:Integer}, Δnoutfactors::Vector{<:Maybe{<:Integer}}, nins, Δninfactors, select = select_start)


    mask = ismissing.(Δnoutfactors)
    # For this case we will just decrease the degrees of freedom, right?
    !any(mask) && return missing

    # TODO: This should just be replaced by zeros to avoid remapping stuff? I promise to change it back if one ever has a problem with snf getting to large matrices with all-zero columns in them
    # Then it can probably be baked into main method as well...
    Δs = alignfor(nouts, all(mask) ? missing : collect(skipmissing(Δnoutfactors)), nins, Δninfactors, select)

    ismissing(Δs) && return missing

    indmap = vcat(findall(.!mask), length(mask) .+ (1:length(nins)))
    ret = zeros(Int, length(nouts) + length(nins))
    ret[indmap] = Δs
    return ret
end

# Should probably bake this into main method as this adds little value
alignfor(nouts::Vector{<:Integer}, Δnoutfactors::Vector{<:Integer}, nins, Δninfactors::Vector{<:Maybe{<:Integer}}, select = select_start) = alignfor(nouts, Δnoutfactors, nins, collect(skipmissing(replace(Δninfactors, (missing => 0)))), select)

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
