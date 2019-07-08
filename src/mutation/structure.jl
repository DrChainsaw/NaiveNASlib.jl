# Overengineered set of strategy types and structs? Not gonna argue with that, but I do this for fun and sometimes I have a wierd idea of what fun is.
# Also, everything is the fault of size transparent vertices, especially SizeStack and those blasted Δnoutfactors!
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

Base type for strategies for how to align size (`nin`/`nout`) when doing structural mutation.

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

Just sets `nin` of each output to the provided value. Sometimes you just know the answer...
"""
struct ChangeNinOfOutputs <: AbstractAlignSizeStrategy
    Δoutsize
end

"""
    FailAlignSizeError

Throws an error.
"""
struct FailAlignSizeError <: AbstractAlignSizeStrategy end

"""
    FailAlignSizeWarn

Logs warning and then proceeds with the next action.
"""
struct FailAlignSizeWarn <: AbstractAlignSizeStrategy
    andthen
end
FailAlignSizeWarn() = FailAlignSizeWarn(FailAlignSizeRevert())

"""
    FailAlignSizeRevert

Reverts new/removed edges (if any).

In other words, `create_edge!` and `remove_edge!` with `strategy = FailAlignSizeRevert` is a noop.

Note: Only works if input vertex is only input once to output vertex due to lazy coding.
For example, if `vout` has inputs `[v1,v2,v1]` and `remove_edge!(v1, vout, startegy=FailAlignSizeRevert)` is called, funtion will exit with graph in invalid state.
Same if `vout` has inputs `[v1,v2]` and `create_edge!(v1,vout)` is called.
"""
struct FailAlignSizeRevert <: AbstractAlignSizeStrategy end

"""
    AlignSizeBoth

Align sizes by changing both input and output considering any Δfactors.
Fallback to another strategy (default `FailAlignSizeError`) if size change is not possible.
"""
struct AlignSizeBoth <: AbstractAlignSizeStrategy
    fallback
end
AlignSizeBoth() = AlignSizeBoth(FailAlignSizeError())

"""
    DecreaseBigger

Try to align size by decreasing in the direction (in/out) which has the bigger size.
Fallback to another strategy (default `AlignSizeBoth`) if size change is not possible.
"""
struct DecreaseBigger <: AbstractAlignSizeStrategy
    fallback
end
DecreaseBigger() = DecreaseBigger(AlignSizeBoth())

"""
    IncreaseSmaller

Try to align size by increasing in the direction (in/out) which has the smaller size.
Fallback to another strategy (default `DecreaseBigger`) if size change is not possible.
"""
struct IncreaseSmaller <: AbstractAlignSizeStrategy
    fallback
end
IncreaseSmaller() = IncreaseSmaller(DecreaseBigger())

"""
    AdjustToCurrentSize

Adjust `nin` of all `outputs` and/or `nout` of all `inputs` to the current input/output size so that `sum(nin.(inputs(v))) == nout(v)` and so that `all(nin.(outputs(v)) .== nout(v))`.

This is a post-align strategy, i.e it will be applied after a structural change has been made.
"""
struct AdjustToCurrentSize <: AbstractAlignSizeStrategy
    fallback
end
AdjustToCurrentSize() = AdjustToCurrentSize(FailAlignSizeError())

"""
    RemoveStrategy

Strategy for removal of a vertex.

Consists of an `AbstractConnectStrategy` for how to treat inputs and outputs of
the removed vertex and an `AbstractAlignSizeStrategy` for how to align sizes of
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

Removes `v` from the graph by removing it from its `inputs` and `outputs`.

It is possible to supply a strategy for how to 1) reconnect the inputs and outputs
of `v` and 2) align the input and output sizes of the `inputs` and `outputs` of `v`.

Default strategy is to first set `nin==nout` for `v` and then connect all its `inputs`
to all its `outputs`.
"""
function remove!(v::MutationVertex, strategy=RemoveStrategy())
    prealignsizes(strategy.align, v, vx -> vx == v) || return

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

Does connection of `items` to `to` at position `inds` depending on the given strategy.
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

tot_nin(v) = tot_nin(trait(v), v)
tot_nin(t::DecoratingTrait, v) = tot_nin(base(t), v)
tot_nin(::MutationTrait, v) = nin(v)[]
tot_nin(::SizeInvariant, v) = unique(nin(v))[]
tot_nin(::SizeTransparent, v) = sum(nin(v))

# Boilerplate
prealignsizes(s::AbstractAlignSizeStrategy, v, will_rm::Function) = prealignsizes(s, v, v, will_rm)
prealignsizes(s::AbstractAlignSizeStrategy, vin, vout, will_rm) = true

# Failure cases
prealignsizes(::FailAlignSizeError, vin, vout, will_rm) = error("Could not align sizes of $(vin) and $(vout)!")
function prealignsizes(s::FailAlignSizeWarn, vin, vout, will_rm)
    @warn "Could not align sizes of $(vin) and $(vout)!"
    return prealignsizes(s.andthen, vin, vout, will_rm)
end
prealignsizes(::FailAlignSizeRevert, vin, vout, will_rm) = false # No action needed?

# Actual actions
function prealignsizes(s::ChangeNinOfOutputs, vin, vout, will_rm)
    Δnin.(outputs(vin), s.Δoutsize...)
    return true
end

function prealignsizes(s::Union{IncreaseSmaller, DecreaseBigger}, vin, vout, will_rm)
    Δinsize = nout(vin) - tot_nin(vout)
    Δoutsize = -Δinsize

    can_change(Δ, factor::Integer) = Δ % factor == 0
    can_change(Δ, ::Missing) = false

    insize_can_change = all( can_change.(Δinsize, minΔninfactor_if(will_rm(vout), vout)))
    if insize_can_change && proceedwith(s, Δinsize)
        Δnin(vout, Δinsize)
        return true
    end

    outsize_can_change = all( can_change.(Δoutsize, minΔnoutfactor_if(will_rm(vin), vin)))
    if outsize_can_change && proceedwith(s, Δoutsize)
        Δnout(vin, Δoutsize)
        return true
    end

    return prealignsizes(s.fallback, vin, vout, will_rm)
end
proceedwith(::DecreaseBigger, Δ::Integer) = Δ <= 0
proceedwith(::IncreaseSmaller, Δ::Integer) = Δ >= 0

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
    # TODO: Add back?
    s = VisitState{Int}(vout) # Just in case we happen to be inside a transparent vertex
    Δnin(vout, Δs[2], s=s)
    Δnout(vin, Δs[1], s=s)
    return true
end

# Boilerplate
postalignsizes(s::AbstractAlignSizeStrategy, v) = postalignsizes(s, v, v)
function postalignsizes(s::AbstractAlignSizeStrategy, vin, vout) end
postalignsizes(s::AdjustToCurrentSize, vin, vout) = postalignsizes(s, vin, vout, trait(vout))
postalignsizes(s::AdjustToCurrentSize, vin, vout, t::DecoratingTrait) = postalignsizes(s, vin, vout, base(t))

# Failure cases
postalignsizes(::FailAlignSizeError, vin, vout) = error("Could not align sizes of $(vin) and $(vout)!")
function postalignsizes(s::FailAlignSizeWarn, vin, vout)
     @warn "Could not align sizes of $(vin) and $(vout)!"
     postalignsizes(s.andthen, vin, vout)
 end
 function postalignsizes(s::FailAlignSizeRevert, vin, vout)
     # TODO: Will fail (silently) in case vin is input to vout many times! CBA to iix that edge case now...
     vin ∈ inputs(vout) && return remove_edge!(vin, vout, strategy=NoSizeChange())
     vin ∉ inputs(vout) && return create_edge!(vin, vout, strategy=NoSizeChange())
 end

# Ok, this one actually does something...
function postalignsizes(s::AdjustToCurrentSize, vin, vout, ::SizeStack)

    # At this point we expect to have nout(vout) to be the correct value while nin of the output edges of vout needs to be adjusted. Problem is that there might be Δfactors (I'm really starting to hate them now) or immutable vertices which prevents us from just changing the input size of the output edges to nout(vout) - nin(voo[i]) where voo[i] is output #i of vout.

    # The equations we want to solve are nout(vout) + Δnoutfactor * x = nin(voo[i]) +  Δninfactor[i] * y[i] for all i.
    Δninfactors = minΔninfactor_only_for.(outputs(vout))
    nins = zeros(Int, length(Δninfactors))
    # Also annoying: We obviously can't use nout(vout) as it is the right size. Instead, we must look at each output(vout), voo[i] and find which input index j of voo[i] is vout and select nin(voo[i])[j]
    let # Let block just to not have temp variable mask laying around...
    # This does the same thing as the below in one line, but is incomprehensible: mapfoldl(vo -> (inputs(vo), nin(vo)),(a,b)-> vcat(a[2],b[2][b[1] .== vout]), outputs(vout))
        mask = mapfoldl(inputs, vcat, outputs(vout)) .== vout
        nins = mapfoldl(nin, vcat, outputs(vout))[mask]
    end

    # What this!? Well, we do want to change vin and vout only, but perhaps this is not possible for one reason or the other (e.g. vin is immutable).
    # Therefore we will try to change each input to vout, starting with vin, until we find one which succeeds.

    # Try vin first if it is one of the inputs to vout. No reason really apart from that it just "feels" more natural to change it over some other vertex which just happens to be input to vout
    vins = sort(inputs(vout), by = voi -> voi == vin ? 0 : indexin([voi], inputs(vout))[])

    function alignsize(cnt=1)
        # Base case: We have tried all inputs and no possible solution was found
        cnt > length(vins) && return missing

        Δnoutfactors =minΔnoutfactor_only_for.(vins[cnt])

        select(f,k) = increase_until(all_positive, f, k)
        Δs = alignfor(nout(vout), Δnoutfactors, nins, Δninfactors, select)

        return ismissing(Δs) ? alignsize(cnt+1) : (Δs, cnt, cnt)
    end

    # TODO: This function and the one above can probably be merged
    # Obstacles:
    #     increase_until only works when Δninfactors are missing and there is only one nout
    #     Search order? First try all nins and then try combinations of them?
    #       -Doesn't it do that already?
    # Also: Why does solver sometimes fail when adding more inputs? Shouldn't it just set them to zero if they are not usable?
    function alignsize_all_inputs(start=1, stop=min(2, length(vins)))
        # Base case: We have tried all options we intend to try with and no possible solution was found
        max(start, stop) > length(vins) && return missing
        Δnoutfactors::Vector{Maybe{Int}} = minΔnoutfactor_only_for.(vins[start:stop])

        function select(f, k, first=true)
            Δs = f(k)
            any(-Δs[1:stop-start+1] .>= nout.(vins[start:stop])) && return f(k .- 1)
            return Δs
        end
        Δs = alignfor(nout(vout), Δnoutfactors, nins, Δninfactors, select)

        if ismissing(Δs)
            return alignsize_all_inputs(start+1, max(1, stop-1))
        elseif any(-Δs[1:stop-start+1] .>= nout.(vins[start:stop]))
            return alignsize_all_inputs(start, stop+1)
        end
        return Δs, start, stop
    end

    res = any(ismissing.(Δninfactors)) ? alignsize_all_inputs() : alignsize()
    ismissing(res) && return postalignsizes(s.fallback, vin, vout)

    Δs, start, stop = res

    for i in start:stop

        Δ = Δs[i-start+1]
        Δ == 0 && continue # To avoid having to support Δnout/Δnin with Δ = 0 for immutable vertices

        s = VisitState{Int}(vins[i])
        visited_in!.(s, outputs(vout))

        Δnout(vins[i], Δ, s=s)
    end

    for (i, voo) in enumerate(outputs(vout))
        Δ = Δs[i+stop-start+1]
        Δ == 0 && continue

        Δvec = Vector{Maybe{Int}}(missing, length(inputs(voo)))
        Δvec[inputs(voo) .== vout] .= Δ

        # Dont touch the parts which already have the correct size
        s = VisitState{Int}(voo, Δvec...)
        visited_out!(s, vout)

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
    alignfor(nouts, Δnoutfactors, nins, Δninfactors, [select])

Returns `Δ` so that `vcat(nouts, nins) .+ Δ |> unique |> length == 1` and so that `all(Δ .% vcat(Δnoutfactors, Δninfactors) .== 0)`.

In other words, return the `Δ` which makes `nouts` equal to all `nins` while still being evenly divisible by the `Δfactors`.

Solves the following system of linear diophantine equations:
```text
nouts[0] + Δnoutfactors[0]*x0 + nouts[1] + Δnoutfactors[1]*x1 + ... = nins[0] + Δninfactors[0] * y0
nouts[0] + Δnoutfactors[0]*x0 + nouts[1] + Δnoutfactors[1]*x1 + ... = nins[1] + Δninfactors[1] * y1
...
```
where `Δ = [x0, x1, ..., y0, y1, ...]`

Argument `select` is a strategy for selecting which solution to use in case the system is underdetermined.

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
function alignfor(nouts, Δnoutfactors, nins, Δninfactors, select = select_start)

    Δnoutfactors_nm = replace(Δnoutfactors, missing => 0)
    Δninfactors_nm = replace(Δninfactors, missing => 0)

    # System is
    # A:
    # Δnoutfactors -Δninfactors[0] 0               0 ...
    # Δnoutfactors  0              -Δninfactors[1] 0 ...
    # ...
    # B:
    # nins .- nouts
    n = length(nins)
    A = hcat(repeat(Δnoutfactors_nm', n), -Diagonal(Δninfactors_nm))

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

    f = k -> (VT + VK * k) .* vcat(Δnoutfactors_nm, Δninfactors_nm)

    return select(f, K)
end

alignfor(nout::Integer, Δnoutfactor::Integer, nins, Δninfactors, select = (f, k) -> increase_until(all_positive, f, k)) = alignfor([nout], [Δnoutfactor], nins,  Δninfactors, select)

# Only for better default for select
alignfor(nout::Integer, ::Missing, nins, Δninfactors, select = select_start) = alignfor(nout, 0, nins, Δninfactors, select)


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

"""
    create_edge!(from::AbstractVertex, to::AbstractVertex; [pos], [strategy])

Create and edge from `from` to `to` at index `pos` in `inputs(to)`.

Sizes will be adjusted based on given `strategy`.
"""
function create_edge!(from::AbstractVertex, to::AbstractVertex; pos = length(inputs(to))+1, strategy = default_create_edge_strat(to))

    prealignsizes(strategy, from, to, v -> false) || return

    push!(outputs(from), to) # Order should never matter for outputs
    insert!(inputs(to), pos, from)

    add_input!(op(to), pos, nout(from))
    add_output!(op(to), trait(to), nout(from))

    postalignsizes(strategy, from, to)
end

default_create_edge_strat(v::AbstractVertex) = default_create_edge_strat(trait(v),v)
default_create_edge_strat(t::DecoratingTrait,v) = default_create_edge_strat(base(t),v)
default_create_edge_strat(::SizeStack,v) = AdjustToCurrentSize()
default_create_edge_strat(::SizeInvariant,v) = IncreaseSmaller()
default_create_edge_strat(::SizeAbsorb,v) = NoSizeChange()

function add_input!(::MutationOp, pos, size) end
add_input!(s::IoSize, pos, size) = insert!(s.nin, pos, size)
add_input!(s::IoIndices, pos, size) = insert!(s.in, pos, collect(1:size+1))
function add_input!(s::IoChange, pos, size)
    add_input!(s.size, pos, 0)
    add_input!(s.indices, pos, 0)
    insert!(s.inΔ, pos, size)
end

function add_output!(::MutationOp, t::MutationTrait, size) end
add_output!(s::MutationOp, t::DecoratingTrait, size) = add_output!(s, base(t), size)
add_output!(s::IoSize, ::SizeStack, size) = Δnout(s, size)
add_output!(s::IoIndices, ::SizeStack, size) = Δnout(s, vcat(s.out, (length(s.out):length(s.out)+size)))
add_output!(s::IoChange, t::SizeStack, size) = Δnout(s, size)


"""
    remove_edge!(from::AbstractVertex, to::AbstractVertex; [nr], [strategy])

Remove edge from `from` to `to`.

If there are multiple edges from `from` to `to` then `nr` can be used to distinguish which one shall be removed.

Sizes will be adjusted based on given `strategy`.
"""
function remove_edge!(from::AbstractVertex, to::AbstractVertex; nr = 1, strategy = default_remove_edge_strat(to))

    prealignsizes(strategy, from, to, v -> false) || return

    in_inds = findall(vx -> vx == from, inputs(to))[nr]
    out_inds =findall(vx -> vx == to, outputs(from))[nr]
    deleteat!(inputs(to), in_inds)
    deleteat!(outputs(from), out_inds)

    rem_input!(op(to), in_inds...)
    add_output!(op(to), trait(to), -nout(from))

    postalignsizes(strategy, from, to)
end

default_remove_edge_strat(v::AbstractVertex) = default_remove_edge_strat(trait(v),v)
default_remove_edge_strat(t::DecoratingTrait,v) = default_remove_edge_strat(base(t),v)
default_remove_edge_strat(::SizeStack,v) = AdjustToCurrentSize()
default_remove_edge_strat(::SizeInvariant,v) = NoSizeChange()
default_remove_edge_strat(::SizeAbsorb,v) = NoSizeChange()

function rem_input!(::MutationOp, pos...) end
rem_input!(s::IoSize, pos...) = deleteat!(s.nin, collect(pos))
rem_input!(s::IoIndices, pos...) = deleteat!(s.in, collect(pos))
function rem_input!(s::IoChange, pos...)
    rem_input!(s.size, pos...)
    rem_input!(s.indices, pos...)
    deleteat!(s.inΔ, collect(pos))
end
