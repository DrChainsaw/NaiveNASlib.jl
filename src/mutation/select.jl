
# Methods to help select or add a number of outputs given a new size as this problem apparently belongs to the class of FU-complete problems. And yes, I curse the day I conceived the idea for this project right now...

"""
    AbstractSelectionStrategy

Base type for how to select the exact inputs/outputs indices from a vertex given a size change.
"""
abstract type AbstractSelectionStrategy end

"""
    LogSelection <: AbstractSelectionStrategy
    LogSelection(msgfun::Function, andthen)
    LogSelection(level, msgfun::Function, andthen)

Logs output from function `msgfun` at LogLevel `level` and executes `AbstractSelectionStrategy andthen`.
"""
struct LogSelection <: AbstractSelectionStrategy
    level::Logging.LogLevel
    msgfun
    andthen::AbstractSelectionStrategy
end
LogSelection(msgfun::Function, andthen) = LogSelection(Logging.Info, msgfun, andthen)
LogSelectionFallback(nextstr, andthen; level=Logging.Warn) = LogSelection(level, v -> "Selection for vertex $(name(v)) failed! $nextstr", andthen)

"""
    SelectionFail <: AbstractSelectionStrategy

Throws an error.
"""
struct SelectionFail <: AbstractSelectionStrategy end

"""
    NoutRevert <: AbstractSelectionStrategy
    NoutRevert()

Reverts output size change for a vertex.
"""
struct NoutRevert <: AbstractSelectionStrategy end

"""
    SelectDirection <: AbstractSelectionStrategy
    SelectDirection()
    SelectDirection(s::AbstractSelectionStrategy)

Select indices for a vertex using `AbstractSelectionStrategy s` (default `OutSelect{Exact}`) in only the direction(s) in which the vertex has changed size.

Intended use it to reduce the number of constraints for a `AbstractJuMPSelectionStrategy` as only the parts of the graph which are changed will be considered.
"""
struct SelectDirection <: AbstractSelectionStrategy
    strategy::AbstractSelectionStrategy
end
SelectDirection() = SelectDirection(OutSelectExact())

"""
    ApplyAfter <: AbstractSelectionStrategy
    ApplyAfter()
    ApplyAfter(s::AbstractSelectionStrategy)
    ApplyAfter(apply::Function, s::AbstractSelectionStrategy)

Invokes `apply(v)` for each `AbstractVertex v` which was changed as a result of `AbstractSelectionStrategy s`.
"""
struct ApplyAfter <: AbstractSelectionStrategy
    apply::Function
    strategy::AbstractSelectionStrategy
end
ApplyAfter() = ApplyAfter(OutSelectExact())
ApplyAfter(s::AbstractSelectionStrategy) = ApplyAfter(apply_mutation, s)

"""
    AbstractJuMPSelectionStrategy

Base type for how to select the exact inputs/outputs indices from a vertex given a size change using JuMP to handle the constraints.
"""
abstract type AbstractJuMPSelectionStrategy <: AbstractSelectionStrategy end

"""
    DefaultJuMPSelectionStrategy <: AbstractJuMPSelectionStrategy

Default strategy intended to be used when adding some extra constraints or objectives to a model on top of the default.
"""
struct DefaultJuMPSelectionStrategy <: AbstractJuMPSelectionStrategy end

fallback(::AbstractJuMPSelectionStrategy) = SelectionFail()

"""
    OutSelect{T} <: AbstractSelectionStrategy
    OutSelectExact()
    OutSelectRelaxed()
    OutSelect{T}(fallback)

If `T == Exact`, output indices for each vertex `v` are selected with the constraint that `nout(v)` shall not change.

If `T == Relaxed`, output indices for each vertex `v` are selected with the objective to deviate as little as possible from `nout(v)`.

Possible to set a `fallback AbstractSelectionStrategy` should it be impossible to select indices according to the strategy.
"""
struct OutSelect{T} <: AbstractJuMPSelectionStrategy
    fallback::AbstractSelectionStrategy
end
OutSelectExact() = OutSelect{Exact}(LogSelectionFallback("Relaxing size constraint...", OutSelectRelaxed()))
OutSelectRelaxed() = OutSelect{Relaxed}(LogSelectionFallback("Reverting...", NoutRevert()))
fallback(s::OutSelect) = s.fallback

"""
    Δoutputs(g::CompGraph, valuefun::Function)
    Δoutputs(s::AbstractSelectionStrategy, g::CompGraph, valuefun::Function)
    Δoutputs(v::AbstractVertex, valuefun::Function)
    Δoutputs(s::AbstractSelectionStrategy, v::AbstractVertex, valuefun::Function)
    Δoutputs(d::Direction, v::AbstractVertex, valuefun::Function)
    Δoutputs(s::AbstractSelectionStrategy, d::Direction, v::AbstractVertex, valuefun::Function)

Change outputs of all vertices of graph `g` (or graph to which `v` is connected) according to the provided `AbstractSelectionStrategy s` (default `OutSelect{Exact}`).

Return true of operation was successful, false otherwise.

Argument `valuefun` provides a vector `value = valuefun(vx)` for any vertex `vx` in the same graph as `v` where `value[i] > value[j]` indicates that output index `i` shall be preferred over `j` for vertex `vx`.

If provided, `Direction d` will narrow down the set of vertices to evaluate so that only vertices which may change as a result of changing size of `v` are considered.
"""
Δoutputs(g::CompGraph, valuefun) = Δoutputs(OutSelectExact(), g, valuefun)
Δoutputs(s::AbstractSelectionStrategy, g::CompGraph, valuefun) = Δoutputs(s, vertices(g), valuefun)
Δoutputs(v::AbstractVertex, valuefun::Function) = Δoutputs(OutSelectExact(), v, valuefun)
Δoutputs(s::AbstractSelectionStrategy, v::AbstractVertex, valuefun::Function) = Δoutputs(s, all_in_graph(v), valuefun)
function Δoutputs(s::SelectDirection, v::AbstractVertex, valuefun::Function)
    nin_change = nin_org(v) != nin(v)
    nout_change = nout_org(v) != nout(v)
    if nout_change && nin_change
        return Δoutputs(s.strategy, Both(), v, valuefun)
    elseif nout_change
        return Δoutputs(s.strategy, Output(), v, valuefun)
    elseif nin_change
        return Δoutputs(s.strategy, Input(), v, valuefun)
    end
    return true
 end

Δoutputs(d::Direction, v::AbstractVertex, valuefun::Function) = Δoutputs(OutSelectExact(), d, v, valuefun)
Δoutputs(s::AbstractSelectionStrategy, d::Direction, v::AbstractVertex, valuefun::Function) = Δoutputs(s, all_in_Δsize_graph(v, d), valuefun)

function Δoutputs(s::ApplyAfter, vs::AbstractVector{<:AbstractVertex}, valuefun::Function)
    success = Δoutputs(s.strategy, vs, valuefun)
    # This is not 100% safe in all cases, as a strategy can chose to not do anything even in this case (although it probably should)
    foreach(s.apply, filter(v -> nout(v) != nout_org(v) || nin(v) != nin_org(v), vs))
    return success
end

function Δoutputs(s::AbstractSelectionStrategy, vs::AbstractVector{<:AbstractVertex}, valuefun::Function)
    success, ins, outs = solve_outputs_selection(s, vs, valuefun)
    if success
        Δoutputs(ins, outs, vs)
    end
    return success
end

"""
    Δoutputs(ins::Dict outs::Dict, vertices::AbstractVector{<:AbstractVertex})

Set input and output indices of each `vi` in `vs` to `outs[vi]` and `ins[vi]` respectively.
"""
function Δoutputs(ins::Dict, outs::Dict, vs::AbstractVector{<:AbstractVertex})

    for vi in vs
        Δnin(OnlyFor(), vi, ins[vi]...)
        Δnout(OnlyFor(), vi, outs[vi])
    end

    for vi in vs
        after_Δnin(vi, ins[vi]...)
        after_Δnout(vi, outs[vi])
    end
end

function Δnin(::OnlyFor, v) end
function Δnin(::OnlyFor, v, inds::Missing) end
function Δnin(s::OnlyFor, v, inds::AbstractVector{<:Integer}...)
    any(inds .!= [1:insize for insize in nin(v)]) || return
    Δnin(s, trait(v), v, inds)
end

function Δnout(::OnlyFor, v, inds::Missing) end
function Δnout(s::OnlyFor, v, inds::AbstractVector{<:Integer})
    inds == 1:nout(v) && return
    Δnout(s, trait(v), v, inds)
end


function solve_outputs_selection(s::LogSelection, vertices::AbstractVector{<:AbstractVertex}, valuefun::Function)
    @logmsg s.level s.msgfun(vertices[1])
    return solve_outputs_selection(s.andthen, vertices, valuefun)
end

solve_outputs_selection(s::SelectionFail, vertices::AbstractVector{<:AbstractVertex}, valuefun::Function) = error("Selection failed for vertex $(name.(vertices))")

function NaiveNASlib.solve_outputs_selection(s::NoutRevert, vertices::AbstractVector{<:AbstractVertex}, valuefun::Function)
    for v in vertices
        Δnout(NaiveNASlib.OnlyFor(), v, nout_org(v) - nout(v))
        diff = nin_org(v) - nin(v)
        Δnin(NaiveNASlib.OnlyFor(), v, diff...)
    end

    return false, Dict(vertices .=> UnitRange.(1, nout.(vertices))), Dict(vertices .=> map(nins -> UnitRange.(1,nins), nin.(vertices)))
end


"""
    solve_outputs_selection(s::AbstractSelectionStrategy, vertices::AbstractVector{<:AbstractVertex}, valuefun::Function)

Returns a tuple `(success, nindict, noutdict)` where `nindict[vi]` are new input indices and `noutdict[vi]` are new output indices for each vertex `vi` in `vertices`.

The function generally tries to maximize `sum(valuefun(vi) .* selected[vi]) ∀ vi in vertices` where `selected[vi]` is all elements in `noutdict[vi]` larger than 0 (negative values in `noutdict` indicates a new output shall be inserted at that position). This however is up to the implementation of the `AbstractSelectionStrategy s`.

Since selection of outputs is not guaranteed to work in all cases, a flag `success` is also returned. If `success` is `false` then applying the new indices may (and probably will) fail.
"""
function solve_outputs_selection(s::AbstractJuMPSelectionStrategy, vertices::AbstractVector{<:AbstractVertex}, valuefun::Function)
    model = selectmodel(s, vertices, values)

    # The binary variables `outselectvars` tells us which existing output indices to select
    # The binary variables `outinsertvars` tells us where in the result we shall insert -1 where -1 means "create a new output (e.g. a neuron)
    # Thus, the result will consist of all selected indices with possibly interlaced -1s

    # TODO: Edge case when vertices are not all in graph (i.e there is at least one vertex for which its inputs or outputs are not part of vertices) and valuefun returns >= 0 might result in size inconsistency.
    # Either make sure values are > 0 for such vertices or add constraints to them!
    # Afaik this happens when only v's inputs are touched. Cleanest fix might be to separate "touch output" from "touch input". See https://github.com/DrChainsaw/NaiveNASlib.jl/issues/39
    outselectvars = Dict(vertices .=> map(v -> @variable(model, [1:length(valuefun(v))], binary=true), vertices))
    outinsertvars = Dict(vertices .=> map(v -> @variable(model, [1:nout(v)], binary=true), vertices))
    # Optimization: Init outinsertvars as empty and only add if needed: outinsertvars = Dict{eltype(outselectvars).types...}()

    objexpr = @expression(model, objective, 0)
    for v in vertices
        data = (model=model, outselectvars=outselectvars, outinsertvars=outinsertvars, objexpr = objexpr, valuefun = valuefun)
        vertexconstraints!(v, s, data)
        objexpr = selectobjective!(s, v, data)
    end

    @objective(model, Max, objexpr)

    JuMP.optimize!(model)

    !accept(s, model) && return solve_outputs_selection(fallback(s), vertices, valuefun)

    return true, extract_ininds_and_outinds(s, outselectvars, outinsertvars)...

end

accept(::AbstractJuMPSelectionStrategy, model::JuMP.Model) = JuMP.termination_status(model) != MOI.INFEASIBLE && JuMP.primal_status(model) == MOI.FEASIBLE_POINT # Beware: primal_status seems unreliable for Cbc. See MathOptInterface issue #822

selectmodel(::AbstractJuMPSelectionStrategy, v, values) = JuMP.Model(JuMP.optimizer_with_attributes(Cbc.Optimizer, "loglevel"=>0))

# First dispatch on traits to sort out things like immutable vertices
vertexconstraints!(v::AbstractVertex, s::AbstractJuMPSelectionStrategy, data) = vertexconstraints!(trait(v), v, s, data)
vertexconstraints!(t::DecoratingTrait, v, s::AbstractJuMPSelectionStrategy, data) = vertexconstraints!(base(t), v, s, data)
vertexconstraints!(t::MutationSizeTrait, v, s::AbstractJuMPSelectionStrategy, data) = vertexconstraints!(s, t, v, data) # Now dispatch on strategy (and trait)
function vertexconstraints!(::Immutable, v, s::AbstractJuMPSelectionStrategy, data)
     @constraint(data.model, data.outselectvars[v] .== 1)
     @constraint(data.model, data.outinsertvars[v] .== 0)
 end


function vertexconstraints!(s::AbstractJuMPSelectionStrategy, t::MutationSizeTrait, v, data)
    sizeconstraint!(s, t, v, data)
    compconstraint!(s, v, (data..., vertex=v))
    inoutconstraint!(s, t, v, data)
end


function sizeconstraint!(::OutSelect{Exact}, t, v, data)
    nselect = min(nout(v), nout_org(v))
    ninsert = max(0, nout(v) - nselect)
    @constraint(data.model, sum(data.outselectvars[v]) == nselect)
    @constraint(data.model, sum(data.outinsertvars[v]) == ninsert)
end

function sizeconstraint!(::OutSelect{Exact}, t::SizeStack, v, data)
    nselect = sum(min.(nin_org(v), nin(v)))
    ninsert = sum(max.(0, nin(v) - nin_org(v)))
    @constraint(data.model, sum(data.outselectvars[v]) == nselect)
    @constraint(data.model, sum(data.outinsertvars[v]) == ninsert)
end

function sizeconstraint!(::OutSelect{Relaxed}, t, v, data)
    @constraint(data.model, sum(data.outselectvars[v]) + sum(data.outinsertvars[v])  >= 1)

    # Handle insertions
    # The constraint that either there are no new outputs or the total number of outputs must be equal to the length of outinsertvars is a somewhat unfortunate result of the approach chosen to solve the problem.
    # If not enforced, we will end up in situations where some indices shall neither be selected nor have insertions. For example, the result might say "keep indices 1,2,3 and insert a new output at index 10".
    # If one can come up with a constraint to formulate "no gaps" (such as the gab above) instead of the current approach the chances of finding a feasible soluion would probably increase.
    # Maybe this https://cs.stackexchange.com/questions/12102/express-boolean-logic-operations-in-zero-one-integer-linear-programming-ilp in combination with this https://math.stackexchange.com/questions/2022967/how-to-model-a-constraint-of-consecutiveness-in-0-1-programming?rq=1
    outsel = data.outselectvars[v]
    outins = data.outinsertvars[v]
    model = data.model
    if length(outins) > length(outsel)
        @constraint(data.model, sum(outins) == length(outins) - sum(outsel))
    elseif length(outins) > 0
        @constraint(data.model, outins .== 0)
    end
end

function inoutconstraint!(s, ::SizeAbsorb, v, data) end
function inoutconstraint!(s, t::SizeTransparent, v, data)
    inoutconstraint!(s, t, v, data.model, data.outselectvars)
    inoutconstraint!(s, t, v, data.model, data.outinsertvars)
end

function inoutconstraint!(s, ::SizeStack, v, model, vardict::Dict)
    offs = 1
    var = vardict[v]
    for (i, vi) in enumerate(inputs(v))
        var_i = vardict[vi]
        # Sizes mismatch when vertex/edge was removed (or edge added)
        if nout_org(vi) == nin_org(v)[i]
            @constraint(model, var_i .== var[offs:offs+length(var_i)-1])
        end
        offs += length(var_i)
    end
end

function inoutconstraint!(s, ::SizeInvariant, v, model, vardict::Dict)
    var = vardict[v]
    for (i, vi) in enumerate(inputs(v))
        # Sizes mismatch when vertex/edge was removed (or edge added)
        nout_org(vi) == nout_org(v) || continue
        var_i = vardict[vi]
        @constraint(model, var_i .== var)
    end
end

function selectobjective!(s::AbstractJuMPSelectionStrategy, v, data)
    value = valueobjective!(s, v, data)
    insertlast = insertlastobjective!(s, v, data)
    return @expression(data.model, data.objexpr + value + insertlast)
end

function selectobjective!(s::OutSelect{Relaxed}, v, data)

    # No thought behind scaling other than wanting to have roughly same order of magnitude
    scale = sum(data.valuefun(v))
    sizediff = @expression(data.model, sum(data.outselectvars[v]) + sum(data.outinsertvars[v]) - nout(v))
    sizediffnorm = norm!(ScaleNorm(scale, MaxNormLinear()), data.model, sizediff)

    default = selectobjective!(DefaultJuMPSelectionStrategy(), v, data)
    return @expression(data.model, default - sizediffnorm)
end

valueobjective!(s::AbstractJuMPSelectionStrategy, v, data) = @expression(data.model, sum(data.valuefun(v) .* data.outselectvars[v]))

function insertlastobjective!(s,v,data)
    insvars = data.outinsertvars[v]
    preferend = collect(1:length(insvars))
    return @expression(data.model, 0.05*sum(insvars .* preferend))
end

function extract_ininds_and_outinds(s, outselectvars::Dict, outinsertvars::Dict)
    outinds = Dict([v => extract_inds(s, outselectvars[v], outinsertvars[v]) for v in keys(outselectvars)])
    ininds = Dict([v => getall(outinds, inputs(v)) for v in keys(outselectvars)])
    return ininds, outinds
end

function extract_inds(s::AbstractJuMPSelectionStrategy, selectvars::T, insertvars::T) where T <: AbstractVector{JuMP.VariableRef}
    # insertvar is 1.0 at indices where a new output shall be added and 0.0 where an existing one shall be selected
    result = -round.(Int, JuMP.value.(insertvars))
    selected = findall(xi -> xi > 0, JuMP.value.(selectvars))

    # TODO: Needs investigation
    sum(result) == 0 && return selected

    j = 1
    for i in eachindex(result)
        if result[i] == 0
            result[i] = selected[j]
            j += 1
        end
    end

    return result
end
