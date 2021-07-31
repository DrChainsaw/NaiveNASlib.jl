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
    Δoutsize::Int
end

"""
    FailAlignSizeNoOp <: AbstractAlignSizeStrategy
    FailAlignSizeNoOp()

Don't do any size change and return failure status.

Note that this means that graphs will most likely be left corrupted state if used as a fallback.
"""
struct FailAlignSizeNoOp <: AbstractAlignSizeStrategy end

"""
    SizeAlignFailError <: Exception

Sizes could not be aligned.
"""
struct SizeAlignFailError <: Exception
    msg::String
end
Base.showerror(io::IO, e::SizeAlignFailError) = print(io, "SizeAlignFailError: ", e.msg)


"""
    FailAlignSizeError <: AbstractAlignSizeStrategy
    FailAlignSizeError()

Throws SizeAlignFailError.
"""
struct FailAlignSizeError <: AbstractAlignSizeStrategy end

"""
    FailAlignSizeWarn <: AbstractAlignSizeStrategy
    FailAlignSizeWarn()
    FailAlignSizeWarn(;andthen, msgfun)

Logs warning and then proceeds with the next action.
"""
struct FailAlignSizeWarn{S, F} <: AbstractAlignSizeStrategy
    andthen::S
    msgfun::F
end
FailAlignSizeWarn(;andthen=FailAlignSizeNoOp(), msgfun=(vin,vout) -> "Could not align sizes of $(vin) and $(vout)!") = FailAlignSizeWarn(andthen, msgfun)

"""
    AlignSizeBoth <: AbstractAlignSizeStrategy
    AlignSizeBoth()
    AlignSizeBoth(fallback)

Align sizes by changing both input and output considering any Δfactors.
Fallback to another strategy (default `FailAlignSizeError`) if size change is not possible.
"""
struct AlignSizeBoth{S, F} <: AbstractAlignSizeStrategy
    fallback::S
    mapstrat::F
end
AlignSizeBoth(;fallback=FailAlignSizeError(),mapstrat=identity) = AlignSizeBoth(fallback, mapstrat)

"""
    DecreaseBigger <: AbstractAlignSizeStrategy
    DecreaseBigger()
    DecreaseBigger(fallback)

Try to align size by decreasing in the direction (in/out) which has the bigger size.
Fallback to another strategy (default `AlignSizeBoth`) if size change is not possible.
"""
struct DecreaseBigger{S, F} <: AbstractAlignSizeStrategy
    fallback::S
    mapstrat::F
end
DecreaseBigger(;fallback=AlignSizeBoth(),mapstrat=identity) = DecreaseBigger(fallback, mapstrat)

"""
    IncreaseSmaller <: AbstractAlignSizeStrategy
    IncreaseSmaller()
    IncreaseSmaller(fallback)

Try to align size by increasing in the direction (in/out) which has the smaller size.
Fallback to another strategy (default `DecreaseBigger`) if size change is not possible.
"""
struct IncreaseSmaller{S,F} <: AbstractAlignSizeStrategy
    fallback::S
    mapstrat::F
end
IncreaseSmaller(;fallback=DecreaseBigger(),mapstrat=identity) = IncreaseSmaller(fallback, mapstrat)

"""
    CheckNoSizeCycle <: AbstractAlignSizeStrategy
    CheckNoSizeCycle()
    CheckNoSizeCycle(;ifok, ifnok)

Check if a size change in one direction causes a change in the other direction and execute strategy `ifnok` (default `FailAlignSizeWarn`) if this is the case.
Motivation is that removing will result in the computation graph being in an invalid state as one of the vertices must fulfill the impossible criterion `nout(v) == nout(v) + a` where `a > 0`.

If no such cycle is detected, then proceed to execute strategy `ifok` (default `IncreaseSmaller`).

Will execute strategy `ifok` if vertex shall not to be removed.
"""
struct CheckNoSizeCycle{S1,S2} <: AbstractAlignSizeStrategy
    ifok::S1
    ifnok::S2
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
struct CheckCreateEdgeNoSizeCycle{S1, S2} <: AbstractAlignSizeStrategy
    ifok::S1
    ifnok::S2
end
CheckCreateEdgeNoSizeCycle(;ifok=IncreaseSmaller(), ifnok=FailAlignSizeWarn(msgfun = (vin,vout) -> "Can not add edge between $(vin) and $(vout)! Size cycle detected!")) = CheckCreateEdgeNoSizeCycle(ifok, ifnok)

"""
    CheckAligned <:AbstractAlignSizeStrategy
    CheckAligned()
    CheckAligned(ifnot)

Check if sizes are already aligned before making a change and return "go ahead" (`true`) if this is the case.
If not, proceed to execute another strategy (default `CheckNoSizeCycle`).
"""
struct CheckAligned{S} <:AbstractAlignSizeStrategy
    ifnot::S
end
CheckAligned() = CheckAligned(CheckNoSizeCycle())

"""
    PostAlign <: AbstractAlignSizeStrategy
    PostAlign()
    PostAlign(s::AbstractAlignSizeStrategy)
    PostAlign(s::AbstractAlignSizeStrategy, fallback)

Align sizes using a `AbstractΔSizeStrategy`.

This is a post-align strategy, i.e it will be applied after a structural change has been made.
"""
struct PostAlign{S,F} <: AbstractAlignSizeStrategy
    sizestrat::S
    fallback::F
end
PostAlign() = PostAlign(DefaultJuMPΔSizeStrategy())
PostAlign(s::AbstractJuMPΔSizeStrategy; fallback = FailAlignSizeError()) = PostAlign(AlignNinToNout(s, ΔSizeFailNoOp()), fallback)
PostAlign(s::AlignNinToNout; fallback=FailAlignSizeError()) = PostAlign(s, fallback)
PostAlign(s; fallback=FailAlignSizeError()) = PostAlign(s, fallback)

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
struct RemoveStrategy{SC<:AbstractConnectStrategy, SA<:AbstractAlignSizeStrategy}
    reconnect::SC
    align::SA
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
    prealignsizes(strategy.align, v, vx -> vx == v) || return false

    undo_fun = remove_with_undo!(v, strategy)

    # Note, success from prealignsizes strategies is implicit as we return if it fails
    success = postalignsizes(strategy.align, v)
    
    if !success
        undo_fun()
    end
    return success
end

remove_with_undo!(v) = remove_with_undo!(v, RemoveStrategy(ConnectNone()))
function remove_with_undo!(v, strategy)
    undo_fun = create_remove_undo(strategy.align, strategy.reconnect, v)

    remove!(v, inputs, outputs, strategy.reconnect)
    remove!(v, outputs, inputs, strategy.reconnect)
    return undo_fun
end


function create_remove_undo(sa, sc, v)
    undo_rm_inputs = create_remove_undo_one_direction(v, inputs, outputs)
    undo_rm_outputs = create_remove_undo_one_direction(v, outputs, inputs)
    return function() 
        undo_rm_inputs() 
        undo_rm_outputs()
    end
end

## Helper function to avoid code duplication. I don't expect this to be able to do
## anything useful unless f1=inputs and f2=outputs or vise versa.
function create_remove_undo_one_direction(v::AbstractVertex, f1, f2)
    prev = copy(f1(v))
    prevf1f2 = Dict(vf1 => copy(f2(vf1)) for vf1 in prev)
    return function()
        set_edges!(f1, v => prev)
        foreach(vf1f2 -> set_edges!(f2, vf1f2), prevf1f2)
    end
end

function set_edges!(f, (v, newins)::Pair)
    empty!(f(v))
    append!(f(v), newins)
end

## Helper function to avoid code duplication. I don't expect this to be able to do
## anything useful unless f1=inputs and f2=outputs or vise versa.
function remove!(v::MutationVertex, f1::Function, f2::Function, s::AbstractConnectStrategy)
    for f1_v in f1(v)
        f1_f2_v = f2(f1_v)
        inds = findall(vx -> vx == v, f1_f2_v)
        deleteat!(f1_f2_v, inds)
        # Connect f2 of v to f2 of v1, e.g connect the inputs to v as inputs to v1 if f2 == inputs
        connect!(s, f1_f2_v, inds, f2(v))
    end
end


"""
    connect!(v, to, inds, items, ::AbstractConnectStrategy)

Does connection of `items` to `to` at position `inds` depending on the given strategy.
"""
#Not sure broadcasting insert! like this is supposed to work, but it does...
connect!(::ConnectAll, to, inds, items) = insert!.(Ref(to), inds, items)
function connect!(::ConnectNone, args...) end

tot_nin(v::AbstractVertex) = tot_nin(trait(v), v)
tot_nin(t::DecoratingTrait, v) = tot_nin(base(t), v)
tot_nin(::SizeInvariant, v) = length(unique(nin(v))) == 1 ? unique(nin(v))[] : nothing
tot_nin(::SizeTransparent, v) = sum(nin(v))
tot_nin(t::MutationTrait, v) = tot_nin(t, base(v))
tot_nin(::MutationTrait, v::CompVertex) = tot_nin(v.computation)
tot_nin(::MutationTrait, v::InputSizeVertex) = nothing
tot_nin(f) = nin(f)[1]

# Boilerplate
prealignsizes(s::AbstractAlignSizeStrategy, v, will_rm::Function) = prealignsizes(s, v, v, will_rm)
prealignsizes(s::AbstractAlignSizeStrategy, vin, vout, will_rm) = true

# Failure cases
prealignsizes(::FailAlignSizeError, vin, vout, will_rm) = throw(SizeAlignFailError("Could not align sizes of $(nameorrepr(vin)) and $(nameorrepr(vout))!"))
function prealignsizes(s::FailAlignSizeWarn, vin, vout, will_rm)
    @warn s.msgfun(vin, vout)
    return prealignsizes(s.andthen, vin, vout, will_rm)
end
prealignsizes(::FailAlignSizeNoOp, vin, vout, will_rm) = false

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
    Δsize!(ΔNout{Exact}(vin, s.Δoutsize;fallback=ΔSizeFailNoOp()), all_in_Δsize_graph(vin, Output()))
    return nout(vin) == expected
end

function prealignsizes(s::Union{IncreaseSmaller, DecreaseBigger}, vin, vout, will_rm)
    Δinsize = nout(vin) - tot_nin(vout)
    Δoutsize = -Δinsize

    strat = proceedwith(s, Δinsize) ? Δninstrat(vout, Δinsize) : Δnoutstrat(vin, Δoutsize)
    success = prealignsizes(s.mapstrat(strat), vin, vout)

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
Δninstrat(::SizeTransparent, v, Δ) = ΔNoutExact(v => Δ; fallback=ΔSizeFailNoOp())
function Δninstrat(::MutationSizeTrait, v, Δ) 
    Δs = Δnin2Δnout(v => Δ)
    Δs[v] = 0
    ΔNoutExact(Δs;fallback=ΔSizeFailNoOp())
end

Δnoutstrat(v, Δ) = Δnoutstrat(trait(v), v, Δ)
Δnoutstrat(t::DecoratingTrait, v, Δ) = Δnoutstrat(base(t), v, Δ)
Δnoutstrat(::Immutable, v, Δ) = ΔSizeFailNoOp()
Δnoutstrat(::SizeTransparent, v, Δ) = ΔNoutExact(v => Δ; fallback=ΔSizeFailNoOp())
Δnoutstrat(::MutationSizeTrait, v, Δ) = ΔNoutExact(v => Δ, (inputs(v) .=> 0)...; fallback=ΔSizeFailNoOp())

function prealignsizes(s::AlignSizeBoth, vin, vout, will_rm)

    strat = AlignNinToNoutVertices(vin, vout, 1:length(nin(vout)), AlignNinToNout(), ΔSizeFailNoOp())
    success = prealignsizes(s.mapstrat(strat), vin, vout)

    if !success
        return prealignsizes(s.fallback, vin, vout, will_rm)
    end
    return true
end

function prealignsizes(s::AbstractΔSizeStrategy, vin, vout)
    vin_all = all_in_Δsize_graph(vin, Output())
    vout_all = all_in_Δsize_graph(vout, Input())
    return Δsize!(s, union(vin_all, vout_all))
end

# Boilerplate
postalignsizes(s::AbstractAlignSizeStrategy, v) = postalignsizes(s, v, v, missing)
postalignsizes(s::AbstractAlignSizeStrategy, vin, vout, pos) = true

# Failure cases
postalignsizes(::FailAlignSizeNoOp, vin, vout, pos) = false
postalignsizes(::FailAlignSizeError, vin, vout, pos) = throw(SizeAlignFailError("Could not align sizes of $(nameorrepr(vin)) and $(nameorrepr(vout))!"))
function postalignsizes(s::FailAlignSizeWarn, vin, vout, pos)
    @warn s.msgfun(vin, vout)
    return postalignsizes(s.andthen, vin, vout, pos)
end

function postalignsizes(s::CheckCreateEdgeNoSizeCycle, vin, vout, pos)
    is_cyclic(ΔnoutSizeGraph(vin)) && return postalignsizes(s.ifnok, vin, vout, pos)
    return postalignsizes(s.ifok, vin, vout, pos)
end

# Ok, this one actually does something...
function postalignsizes(s::PostAlign, vin, vout, pos)
    vin_all = all_in_Δsize_graph(vin, Output())
    vout_all = all_in_Δsize_graph(vout, Input())

    success = Δsize!(s.sizestrat, union(vin_all, vout_all))
    if !success
        return postalignsizes(s.fallback, vin, vout, pos)
    end
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
function Base.insert!(vin::AbstractVertex, factory, outselect=identity)

    prevouts = copy(outselect(outputs(vin)))
    deleteat!(outputs(vin), findall(vx -> vx in prevouts, outputs(vin)))

    vnew = factory(vin)
    for vout in prevouts
        inds = findall(vx -> vx == vin, inputs(vout))
        inputs(vout)[inds] .= vnew
        push!(outputs(vnew), vout)
    end
    return true # TODO: Check sizes and try to align if not ok?
end

"""
    create_edge!(from::AbstractVertex, to::AbstractVertex; [pos], [strategy])

Create and edge from `from` to `to` at index `pos` in `inputs(to)`.

Sizes will be adjusted based on given `strategy`.
"""
function create_edge!(from::AbstractVertex, to::AbstractVertex; pos = length(inputs(to))+1, strategy = default_create_edge_strat(to))

    prealignsizes(strategy, from, to, v -> false) || return false

    push!(outputs(from), to) # Order should never matter for outputs
    insert!(inputs(to), pos, from)

    # Note, success from prealignsizes strategies is implicit as we return if it fails
    success = postalignsizes(strategy, from, to, pos)

    # Revert what we did if failed
    if !success
        deleteat!(outputs(from), length(outputs(from)))
        deleteat!(inputs(to), pos)
    end
    return success
end

default_create_edge_strat(v::AbstractVertex) = default_create_edge_strat(trait(v),v)
default_create_edge_strat(t::DecoratingTrait,v) = default_create_edge_strat(base(t),v)
default_create_edge_strat(::SizeInvariant,v) = CheckCreateEdgeNoSizeCycle(ifok=AlignSizeBoth())
default_create_edge_strat(::SizeAbsorb,v) = NoSizeChange()
function default_create_edge_strat(::SizeStack,v)
    alignstrat = TruncateInIndsToValid(AlignNinToNout(DefaultJuMPΔSizeStrategy(), ΔSizeFailNoOp()))
    CheckCreateEdgeNoSizeCycle(ifok=PostAlign(alignstrat))
end

"""
    remove_edge!(from::AbstractVertex, to::AbstractVertex; [nr], [strategy])

Remove edge from `from` to `to`.

If there are multiple edges from `from` to `to` then `nr` can be used to distinguish which one shall be removed.

Sizes will be adjusted based on given `strategy`.
"""
function remove_edge!(from::AbstractVertex, to::AbstractVertex; nr = 1, strategy = default_remove_edge_strat(to))

    prealignsizes(strategy, from, to, v -> false) || return false

    in_ind  = findall(vx -> vx == from, inputs(to))[nr]
    out_ind = findall(vx -> vx == to, outputs(from))[1] #which output should not matter

    deleteat!(inputs(to), in_ind)
    deleteat!(outputs(from), out_ind)

    # Note, success from prealignsizes strategies is implicit as we return if it fails
    success = postalignsizes(strategy, from, to, in_ind)

    # Revert what we did if failed
    if !success
        insert!(inputs(to), in_ind, from)
        insert!(outputs(from), out_ind, to)
    end
    return success
end

default_remove_edge_strat(v::AbstractVertex) = default_remove_edge_strat(trait(v),v)
default_remove_edge_strat(t::DecoratingTrait,v) = default_remove_edge_strat(base(t),v)
default_remove_edge_strat(::SizeStack,v) = PostAlign()
default_remove_edge_strat(::SizeInvariant,v) = NoSizeChange()
default_remove_edge_strat(::SizeAbsorb,v) = NoSizeChange()
