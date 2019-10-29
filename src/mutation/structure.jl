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
    NoSizeChange <: AbstractAlignSizeStrategy
    NoSizeChange()

Don't do any size change.
"""
struct NoSizeChange <: AbstractAlignSizeStrategy end

"""
    ChangeNinOfOutputs <: AbstractAlignSizeStrategy
    ChangeNinOfOutputs(Δoutsize)

Just sets `nin` of each output to the provided value. Sometimes you just know the answer...
"""
struct ChangeNinOfOutputs <: AbstractAlignSizeStrategy
    Δoutsize
end

"""
    FailAlignSizeError <: AbstractAlignSizeStrategy
    FailAlignSizeError()

Throws an error.
"""
struct FailAlignSizeError <: AbstractAlignSizeStrategy end

"""
    FailAlignSizeWarn <: AbstractAlignSizeStrategy
    FailAlignSizeWarn()
    FailAlignSizeWarn(;andthen, msgfun)

Logs warning and then proceeds with the next action.
"""
struct FailAlignSizeWarn <: AbstractAlignSizeStrategy
    andthen
    msgfun
end
FailAlignSizeWarn(;andthen=FailAlignSizeRevert(), msgfun=(vin,vout) -> "Could not align sizes of $(vin) and $(vout)!") = FailAlignSizeWarn(andthen, msgfun)

"""
    FailAlignSizeRevert <: AbstractAlignSizeStrategy
    FailAlignSizeRevert()

Reverts new/removed edges (if any).

In other words, `create_edge!` and `remove_edge!` with `strategy = FailAlignSizeRevert` is a noop.

Note: Only works if input vertex is only input once to output vertex due to lazy coding.
For example, if `vout` has inputs `[v1,v2,v1]` and `remove_edge!(v1, vout, startegy=FailAlignSizeRevert)` is called, funtion will exit with graph in invalid state.
Same if `vout` has inputs `[v1,v2]` and `create_edge!(v1,vout)` is called.
"""
struct FailAlignSizeRevert <: AbstractAlignSizeStrategy end

"""
    AlignSizeBoth <: AbstractAlignSizeStrategy
    AlignSizeBoth()
    AlignSizeBoth(fallback)

Align sizes by changing both input and output considering any Δfactors.
Fallback to another strategy (default `FailAlignSizeError`) if size change is not possible.
"""
struct AlignSizeBoth <: AbstractAlignSizeStrategy
    fallback
end
AlignSizeBoth() = AlignSizeBoth(FailAlignSizeError())

"""
    DecreaseBigger <: AbstractAlignSizeStrategy
    DecreaseBigger()
    DecreaseBigger(fallback)

Try to align size by decreasing in the direction (in/out) which has the bigger size.
Fallback to another strategy (default `AlignSizeBoth`) if size change is not possible.
"""
struct DecreaseBigger <: AbstractAlignSizeStrategy
    fallback
end
DecreaseBigger() = DecreaseBigger(AlignSizeBoth())

"""
    IncreaseSmaller <: AbstractAlignSizeStrategy
    IncreaseSmaller()
    IncreaseSmaller(fallback)

Try to align size by increasing in the direction (in/out) which has the smaller size.
Fallback to another strategy (default `DecreaseBigger`) if size change is not possible.
"""
struct IncreaseSmaller <: AbstractAlignSizeStrategy
    fallback
end
IncreaseSmaller() = IncreaseSmaller(DecreaseBigger())

"""
    SelectOutputs <: AbstractAlignSizeStrategy
    SelectOutputs(;select=OutSelectExact(),align=IncreaseSmaller(), valuefun=v -> ones(nout_org(v)))

First align size using `align`, then select outputs (through `Δoutputs`) using `select` and `valuefun` if size alignment is successful.

Motivation is that selecting outputs is more efficient to do when original sizes are aligned.
"""
struct SelectOutputs <: AbstractAlignSizeStrategy
    selectstrategy::AbstractSelectionStrategy
    alignstrategy::AbstractAlignSizeStrategy
    valuefun::Function
end
SelectOutputs(;select=OutSelectExact(),align=IncreaseSmaller(), valuefun=v -> ones(nout_org(v))) = SelectOutputs(select, align, valuefun)

"""
    PostSelectOutputs <: AbstractAlignSizeStrategy
    PostSelectOutputs(;select=OutSelectExact(),align=IncreaseSmaller(), valuefun=v -> ones(nout_org(v)))

Post change alignment strategy which first aligns size using `align`, then select outputs (through `Δoutputs`) using `select` and `valuefun` if size alignment is successful. If `Δoutputs` is not successful, the `fallback` strategy will be invoked.

Motivation is basically convenience when creating new edges between vertices.
"""
struct PostSelectOutputs <: AbstractAlignSizeStrategy
    selectstrategy::AbstractSelectionStrategy
    alignstrategy::AbstractAlignSizeStrategy
    valuefun::Function
    fallback::AbstractAlignSizeStrategy
end
PostSelectOutputs(;select=OutSelectExact(),align=PostAlignJuMP(), valuefun=v -> ones(nout_org(v)), fallback=FailAlignSizeRevert()) = PostSelectOutputs(select, align, valuefun, fallback)


"""
    ApplyMutation <: AbstractAlignSizeStrategy
    ApplyMutation()
    ApplyMutation(strategy::AbstractAlignSizeStrategy)

First align size using `strategy`, then invoke `apply_mutation` if size alignment is successful.

Motivation is that selecting outputs is more efficient to do when original sizes are aligned.
"""
struct ApplyMutation <: AbstractAlignSizeStrategy
    strategy::AbstractAlignSizeStrategy
end
ApplyMutation() = ApplyMutation(SelectOutputs())

"""
    PostApplyMutation <: AbstractAlignSizeStrategy
    PostApplyMutation()
    PostApplyMutation(strategy::AbstractAlignSizeStrategy)

Post change alignment strategy which first aligns size using `strategy`, then invoke `apply_mutation` if size alignment is successful.

Motivation is basically convenience when creating new edges between vertices.
"""
struct PostApplyMutation <: AbstractAlignSizeStrategy
    strategy::AbstractAlignSizeStrategy
end
PostApplyMutation() = PostApplyMutation(PostSelectOutputs())

"""
    CheckNoSizeCycle <: AbstractAlignSizeStrategy
    CheckNoSizeCycle()
    CheckNoSizeCycle(;ifok, ifnok)

Check if a size change in one direction causes a change in the other direction and execute strategy `ifnok` (default `FailAlignSizeWarn`) if this is the case.
Motivation is that removing will result in the computation graph being in an invalid state as one of the vertices must fulfill the impossible criterion `nout(v) == nout(v) + a` where `a > 0`.

If no such cycle is detected, then proceed to execute strategy `ifok` (default `IncreaseSmaller`).

Will execute strategy `ifok` if vertex shall not to be removed.
"""
struct CheckNoSizeCycle <: AbstractAlignSizeStrategy
    ifok
    ifnok
end
CheckNoSizeCycle(;ifok=IncreaseSmaller(), ifnok=FailAlignSizeWarn(msgfun = (vin,vout) -> "Can not remove vertex $(vin)! Size cycle detected!")) = CheckNoSizeCycle(ifok, ifnok)

"""
    CheckCreateEdgeNoSizeCycle <: AbstractAlignSizeStrategy
    CheckCreateEdgeNoSizeCycle()
    CheckCreateEdgeNoSizeCycle(;ifok, ifnok)

Check if adding an edge creates the same type of size cycle that `CheckNoSizeCycle` checks for and execute `ifnok` (default `FailAlignSizeWarn`) if this is the case.
Motivation is that removing will result in the computation graph being in an invalid state as one of the vertices must fulfill the impossible criterion `nout(v) == nout(v) + a` where `a > 0`.

If no such cycle is detected, then proceed to execute strategy `ifok` (default `IncreaseSmaller`).

Will check both at `prealignsizes` (i.e before edge is added) and at `postalignsizes` (i.e after edge is added).
"""
struct CheckCreateEdgeNoSizeCycle <: AbstractAlignSizeStrategy
    ifok
    ifnok
end
CheckCreateEdgeNoSizeCycle(;ifok=IncreaseSmaller(), ifnok=FailAlignSizeWarn(msgfun = (vin,vout) -> "Can not add edge between $(vin) and $(vout)! Size cycle detected!")) = CheckCreateEdgeNoSizeCycle(ifok, ifnok)

"""
    CheckAligned <:AbstractAlignSizeStrategy
    CheckAligned()
    CheckAligned(ifnot)

Check if sizes are already aligned before making a change and return "go ahead" (`true`) if this is the case.
If not, proceed to execute another strategy (default `CheckNoSizeCycle`).
"""
struct CheckAligned <:AbstractAlignSizeStrategy
    ifnot
end
CheckAligned() = CheckAligned(CheckNoSizeCycle())

"""
    PostAlignJuMP <: AbstractAlignSizeStrategy
    PostAlignJuMP()
    PostAlignJuMP(s::AbstractAlignSizeStrategy)
    PostAlignJuMP(s::AbstractAlignSizeStrategy, fallback)

Align sizes using a `AbstractJuMPΔSizeStrategy`.

This is a post-align strategy, i.e it will be applied after a structural change has been made.
"""
struct PostAlignJuMP <: AbstractAlignSizeStrategy
    sizestrat::AbstractJuMPΔSizeStrategy
    fallback
end
PostAlignJuMP() = PostAlignJuMP(DefaultJuMPΔSizeStrategy())
PostAlignJuMP(s::AbstractJuMPΔSizeStrategy; fallback = FailAlignSizeError()) = PostAlignJuMP(AlignNinToNout(s, ΔSizeFailNoOp()), fallback)
PostAlignJuMP(s::AlignNinToNout; fallback=FailAlignSizeError()) = PostAlignJuMP(s, fallback)

"""
    RemoveStrategy
    RemoveStrategy()
    RemoveStrategy(rs::AbstractConnectStrategy)
    RemoveStrategy(as::AbstractAlignSizeStrategy)
    RemoveStrategy(rs::AbstractConnectStrategy, as::AbstractAlignSizeStrategy)

Strategy for removal of a vertex.

Consists of an `AbstractConnectStrategy` for how to treat inputs and outputs of
the removed vertex and an `AbstractAlignSizeStrategy` for how to align sizes of
inputs and outputs.
"""
struct RemoveStrategy
    reconnect::AbstractConnectStrategy
    align::AbstractAlignSizeStrategy
end
RemoveStrategy() = RemoveStrategy(ConnectAll(), CheckAligned())
RemoveStrategy(rs::AbstractConnectStrategy) = RemoveStrategy(rs, CheckAligned())
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

tot_nin(v) = tot_nin(trait(v), v)
tot_nin(t::DecoratingTrait, v) = tot_nin(base(t), v)
tot_nin(::MutationTrait, v) = nin(v)[]
tot_nin(::SizeInvariant, v) = length(unique(nin(v))) == 1 ? unique(nin(v))[] : nothing
tot_nin(::SizeTransparent, v) = sum(nin(v))

# Boilerplate
prealignsizes(s::AbstractAlignSizeStrategy, v, will_rm::Function) = prealignsizes(s, v, v, will_rm)
prealignsizes(s::AbstractAlignSizeStrategy, vin, vout, will_rm) = true

# Failure cases
prealignsizes(::FailAlignSizeError, vin, vout, will_rm) = error("Could not align sizes of $(vin) and $(vout)!")
function prealignsizes(s::FailAlignSizeWarn, vin, vout, will_rm)
    @warn s.msgfun(vin, vout)
    return prealignsizes(s.andthen, vin, vout, will_rm)
end
prealignsizes(::FailAlignSizeRevert, vin, vout, will_rm) = false # No action needed?

# Actual actions
function prealignsizes(s::CheckAligned, vin, vout, will_rm)
    nout(vin) == tot_nin(vout) && return true
    return prealignsizes(s.ifnot, vin, vout, will_rm)
end

function prealignsizes(s::CheckNoSizeCycle, vin, vout, will_rm)
    if will_rm(vout) && vin==vout
        is_cyclic(ΔnoutSizeGraph(vin)) && return prealignsizes(CheckAligned(s.ifnok), vin, vout, will_rm)
    end
    return prealignsizes(s.ifok, vin, vout, will_rm)
end

function prealignsizes(s::CheckCreateEdgeNoSizeCycle, vin, vout, will_rm)
    sg = ΔnoutSizeGraph(vin)
    if vout in keys(sg.metaindex[:vertex])
        add_edge!(sg, sg[vin, :vertex], sg[vout, :vertex])
        is_cyclic(ΔnoutSizeGraph(vin)) && return prealignsizes(CheckAligned(s.ifnok), vin, vout, will_rm)
    end
    return prealignsizes(s.ifok, vin, vout, will_rm)
end

function prealignsizes(s::ChangeNinOfOutputs, vin, vout, will_rm)
    expected = nout(vin) + s.Δoutsize
    Δsize(ΔNout{Exact}(vin, s.Δoutsize, ΔSizeFailNoOp()), all_in_Δsize_graph(vin, Output()))
    return nout(vin) == expected
end

function prealignsizes(s::ApplyMutation, vin, vout, will_rm)
    if prealignsizes(s.strategy, vin, vout, will_rm)
        apply_mutation.(all_in_graph(vin))
        return true
    end
    return false
end

prealignsizes(s::PostApplyMutation, vin, vout, will_rm) = prealignsizes(s.strategy, vin, vout, will_rm)

function prealignsizes(s::SelectOutputs, vin, vout, will_rm)
    if prealignsizes(s.alignstrategy, vin, vout, will_rm)
        return Δoutputs(s.selectstrategy, vin, s.valuefun)
    end
    return false
end

prealignsizes(s::PostSelectOutputs, vin, vout, will_rm) = prealignsizes(s.alignstrategy, vin, vout, will_rm)

function prealignsizes(s::Union{IncreaseSmaller, DecreaseBigger}, vin, vout, will_rm)
    Δinsize = nout(vin) - tot_nin(vout)
    Δoutsize = -Δinsize

    strat = proceedwith(s, Δinsize) ? Δninstrat(vout, Δinsize) : Δnoutstrat(vin, Δoutsize)
    success = prealignsizes(strat, vin, vout)

    if !success
        return prealignsizes(s.fallback, vin, vout, will_rm)
    end
    return true
end
proceedwith(::DecreaseBigger, Δ::Integer) = Δ <= 0
proceedwith(::IncreaseSmaller, Δ::Integer) = Δ >= 0

Δninstrat(v, Δ) = Δninstrat(trait(v), v, Δ)
Δninstrat(t::DecoratingTrait, v, Δ) = Δninstrat(base(t), v, Δ)
Δninstrat(::Immutable, v, Δ) = ΔSizeFailNoOp()
Δninstrat(::MutationSizeTrait, v, Δ) = ΔNin{Exact}(v, [Δ], ΔSizeFailNoOp())
Δninstrat(::SizeTransparent, v, Δ) = ΔNout{Exact}(v, Δ, ΔSizeFailNoOp())

Δnoutstrat(v, Δ) = Δnoutstrat(trait(v), v, Δ)
Δnoutstrat(t::DecoratingTrait, v, Δ) = Δnoutstrat(base(t), v, Δ)
Δnoutstrat(::Immutable, v, Δ) = ΔSizeFailNoOp()
Δnoutstrat(::MutationSizeTrait, v, Δ) = ΔNout{Exact}(v, Δ, ΔSizeFailNoOp())

function prealignsizes(s::AlignSizeBoth, vin, vout, will_rm)

    strat = AlignNinToNoutVertices(vin, vout, 1:length(nin(vout)), AlignNinToNout(), ΔSizeFailNoOp())
    success = prealignsizes(strat, vin, vout)

    if !success
        return prealignsizes(s.fallback, vin, vout, will_rm)
    end
    return true
end

function prealignsizes(s::AbstractΔSizeStrategy, vin, vout)
    vin_all = all_in_Δsize_graph(vin, Output())
    vout_all = all_in_Δsize_graph(vout, Input())

    verts = union(vin_all, vout_all)
    success, nins, nouts = newsizes(s, verts)

    if success
        Δsize(nins, nouts, verts)
        return true
    end
    return false
end

# Boilerplate
postalignsizes(s::AbstractAlignSizeStrategy, v) = postalignsizes(s, v, v)
postalignsizes(s::AbstractAlignSizeStrategy, vin, vout) = true

# Failure cases
postalignsizes(::FailAlignSizeError, vin, vout) = error("Could not align sizes of $(vin) and $(vout)!")
function postalignsizes(s::FailAlignSizeWarn, vin, vout)
    @warn s.msgfun(vin, vout)
    return postalignsizes(s.andthen, vin, vout)
end
function postalignsizes(s::FailAlignSizeRevert, vin, vout)
    n = sum(inputs(vout) .== vin)
    # Can maybe be supported by comparing nin_org(vout) to nout_org(vin): If they match then vin was there before, else it shall be removed?
    @assert n <= 1 "Case when vin is input to vout multiple times not implemented!"

    if n == 1
        remove_edge!(vin, vout, strategy=NoSizeChange())
    else
        create_edge!(vin, vout, strategy=NoSizeChange())
    end
    return false
end

function postalignsizes(s::PostSelectOutputs, vin, vout)
    if postalignsizes(s.alignstrategy, vin, vout)
        vin_all = nout(vin) == nout_org(vin) ? AbstractVertex[] : all_in_Δsize_graph(vin, Output())
        vout_all = nin(vout) == nin_org(vout) ? AbstractVertex[] : all_in_Δsize_graph(vout, Input())

        verts = union(vin_all, vout_all)

        isempty(verts) && return true
        success = Δoutputs(s.selectstrategy, verts, s.valuefun)

        if !success
            return postalignsizes(s.fallback, vin, vout)
        end
        return success
    end
    return false
end

function postalignsizes(s::PostApplyMutation, vin, vout)
    if postalignsizes(s.strategy, vin, vout)
        apply_mutation.(all_in_graph(vin))
        return true
    end
    return false
end

function postalignsizes(s::CheckCreateEdgeNoSizeCycle, vin, vout)
    sg = ΔnoutSizeGraph(vin)
    is_cyclic(ΔnoutSizeGraph(vin)) && return postalignsizes(s.ifnok, vin, vout)
    return postalignsizes(s.ifok, vin, vout)
end

# Ok, this one actually does something...
function postalignsizes(s::PostAlignJuMP, vin, vout)
    vin_all = all_in_Δsize_graph(vin, Output())
    vout_all = all_in_Δsize_graph(vout, Input())

    verts = union(vin_all, vout_all)
    success, nins, nouts = newsizes(s.sizestrat, verts)
    if !success
        return postalignsizes(s.fallback, vin, vout)
    end
    Δsize(nins, nouts, verts)
    return success
end

"""
    insert!(vin::AbstractVertex, factory::Function, outselect::Function=identity)

Replace `vin` as input to all outputs of `vin` with vertex produced by `factory`.

The function `factory` takes `vin` as input.

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
default_create_edge_strat(::SizeStack,v) = CheckCreateEdgeNoSizeCycle(ifok=PostAlignJuMP())
default_create_edge_strat(::SizeInvariant,v) = CheckCreateEdgeNoSizeCycle(ifok=IncreaseSmaller())
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

    add_output!(op(to), trait(to), -nin(to)[in_inds])
    rem_input!(op(to), in_inds...)

    postalignsizes(strategy, from, to)
end

default_remove_edge_strat(v::AbstractVertex) = default_remove_edge_strat(trait(v),v)
default_remove_edge_strat(t::DecoratingTrait,v) = default_remove_edge_strat(base(t),v)
default_remove_edge_strat(::SizeStack,v) = PostAlignJuMP()
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
