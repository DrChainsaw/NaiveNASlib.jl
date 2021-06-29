
"""
    NeuronIndices

Treat vertices as having parameters which represents neurons when formulating the size change problem.
"""
struct NeuronIndices end

# Just another name for Δsize with the corresponding direction
Δnout(v::AbstractVertex, Δ::Integer) = Δsize(Δsizetype(v), Output(), v, Δ)
Δnin(v::AbstractVertex, Δs::Maybe{<:Integer}...) = Δsize(Δsizetype(v), Input(), v, Δs...)

Δnout(valuefun::Function, v::AbstractVertex, Δ::Integer) = Δsize(valuefun, Δsizetype(v), Output(), v, Δ)
Δnin(valuefun::Function, v::AbstractVertex, Δ::Maybe{<:Integer}, Δs::Maybe{<:Integer}...) = Δsize(valuefun, Δsizetype, Input(), v, Δ, Δs...)


Δsizetype(v::AbstractVertex, seen=Set()) = v in seen ? nothing : Δsizetype(trait(v), v, push!(seen, v))
Δsizetype(t::DecoratingTrait, v, seen) = Δsizetype(base(t), v, seen)
Δsizetype(t::MutationTrait, v, seen) = Δsizetype(t, base(v), seen)
Δsizetype(::MutationTrait, ::InputVertex, seen) = nothing
Δsizetype(::MutationTrait, v::CompVertex, seen) = Δsizetype(v.computation)
Δsizetype(f) = NeuronIndices()

Δsizetype(::SizeTransparent, v, seen) = partialsort!(vcat(map(vi -> Δsizetype(vi, seen), inputs(v)), map(vo -> Δsizetype(vo, seen), outputs(v))), 1; by=Δsizetypeprio, rev=true)

Δsizetypeprio(::Nothing) = 0
Δsizetypeprio(::ScalarSize) = 1
Δsizetypeprio(::NeuronIndices) = 2


"""
    Δsize(case, d::Direction, v::AbstractVertex, Δ...)

Change size of `v` by `Δ` in direction `d`.
"""
Δsize(case, d::Input, v, Δs::Maybe{T}...) where T <:Integer = Δsize(case, ΔNinExact(v, collect(Maybe{T}, Δs)), all_in_Δsize_graph(v, d))
Δsize(case, d::Output, v, Δ::T) where T <: Integer = Δsize(case, ΔNoutExact(v, Δ), all_in_Δsize_graph(v, d))

Δsize(f, case, ::Input, v, Δs::Maybe{T}...) where T <:Integer = Δsize(f, case, ΔNinExact(v, collect(Maybe{T}, Δs)), all_in_graph(v))
Δsize(f, case, ::Output, v, Δ::T) where T <: Integer = Δsize(f, case, ΔNoutExact(v, Δ), all_in_graph(v))


"""
    Δsize(valuefun, g::CompGraph)
    Δsize(valuefun, s::AbstractΔSizeStrategy, g::CompGraph)
    Δsize(valuefun, v::AbstractVertex)
    Δsize(valuefun, s::AbstractΔSizeStrategy, v::AbstractVertex)

Change output neurons of all vertices of graph `g` (or graph to which `v` is connected) according to the provided `AbstractΔSizeStrategy s` (default `OutSelect{Exact}`).

Return true of operation was successful, false otherwise.

Argument `valuefun` provides a vector `value = valuefun(vx)` for any vertex `vx` in the same graph as `v` where `value[i] > value[j]` indicates that output neuron index `i` shall be preferred over `j` for vertex `vx`.

If provided, `Direction d` will narrow down the set of vertices to evaluate so that only vertices which may change as a result of changing size of `v` are considered.
"""
Δsize(valuefun, g::CompGraph) = Δsize(valuefun, DefaultJuMPΔSizeStrategy(), g)
Δsize(valuefun, s::AbstractΔSizeStrategy, g::CompGraph) = Δsize(valuefun, s, vertices(g))
Δsize(valuefun, v::AbstractVertex) = Δsize(valuefun, DefaultJuMPΔSizeStrategy(), v)
Δsize(valuefun, s::AbstractΔSizeStrategy, v::AbstractVertex) = Δsize(valuefun, s, all_in_graph(v))
function Δsize(valuefun, s::SelectDirection, v::AbstractVertex)
    nin_change = nin_org(v) != nin(v)
    nout_change = nout_org(v) != nout(v)
    
    d = if nout_change && nin_change
        Both()
    elseif nout_change
        Output()
    elseif nin_change
        Input()
    else
        return true
    end

    return Δsize(valuefun, s.strategy, d, v)
end

Δsize(valuefun, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}) = Δsize(valuefun, partialsort!(Δsizetype.(vs), 1; by=Δsizetypeprio, rev=true), s, vs)

function Δsize(valuefun, case::NeuronIndices, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex})
    success, ins, outs = solve_outputs_selection(s, vs, valuefun)
    if success
        Δsize(case, ins, outs, vs)
    end
    return success
end

function Δsize(valuefun, ::NeuronIndices, s::TruncateInIndsToValid, vs::AbstractVector{<:AbstractVertex})
    success, ins, outs = solve_outputs_selection(s.strategy, vs, valuefun)
    if success
        for (vv, ininds) in ins
            for innr in eachindex(ininds)
                ininds[innr] = aligntomax(nin_org(vv)[innr], ininds[innr])
            end
            newins, newouts = align_outs_to_ins(vv, ins[vv], outs[vv])
            ins[vv] = newins
            outs[vv] = newouts
        end
        Δsize(ins, outs, vs)
    end
    return success
end

aligntomax(maxval, ::Missing) = missing
function aligntomax(maxval, arr)
    arr = copy(arr)
    maxind = argmax(arr)
    while arr[maxind] > maxval
        newval = maxval
        while newval in arr && newval > -1
            newval -= 1
        end
        arr[maxind] = newval == 0 ? -1 : newval
        maxind = argmax(arr)
    end
    return arr
end

align_outs_to_ins(v, ins, outs) = align_outs_to_ins(trait(v), v, ins, outs)
align_outs_to_ins(t::DecoratingTrait, v, ins, outs) = align_outs_to_ins(base(t), v, ins, outs)
align_outs_to_ins(t, v, ins, outs) = ins,outs
function align_outs_to_ins(::SizeInvariant, v, ins::AbstractArray, outs)
    isempty(ins) && return ins, outs
    inds = ins[1]
    newins = repeat([inds], length(ins))
    newouts = ismissing(outs) ? outs : inds
    return newins, newouts
end

"""
    Δsize(ins::Dict outs::Dict, vertices::AbstractVector{<:AbstractVertex})

Set input and output indices of each `vi` in `vs` to `outs[vi]` and `ins[vi]` respectively.
"""
function Δsize(::NeuronIndices, ins::AbstractDict, outs::AbstractDict, vs::AbstractVector{<:AbstractVertex})

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


function solve_outputs_selection(s::LogΔSizeExec, vertices::AbstractVector{<:AbstractVertex}, valuefun::Function)
    @logmsg s.level s.msgfun(vertices[1])
    return solve_outputs_selection(s.andthen, vertices, valuefun)
end

solve_outputs_selection(::ΔSizeFailError, vertices::AbstractVector{<:AbstractVertex}, valuefun::Function) = error("Size change failed for vertex $(name.(vertices))")

"""
    solve_outputs_selection(s::AbstractΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex}, valuefun::Function)

Returns a tuple `(success, nindict, noutdict)` where `nindict[vi]` are new input neuron indices and `noutdict[vi]` are new output neuron indices for each vertex `vi` in `vertices`.

The function generally tries to maximize `sum(valuefun(vi) .* selected[vi]) ∀ vi in vertices` where `selected[vi]` is all elements in `noutdict[vi]` larger than 0 (negative values in `noutdict` indicates a new output shall be inserted at that position). This however is up to the implementation of the `AbstractΔSizeStrategy s`.

Since selection of outputs is not guaranteed to work in all cases, a flag `success` is also returned. If `success` is `false` then applying the new indices may (and probably will) fail.
"""
function solve_outputs_selection(s::AbstractJuMPΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex}, valuefun::Function)
    model = selectmodel(NeuronIndices(), s, vertices, values)

    # The binary variables `outselectvars` tells us which existing output indices to select
    # The integer variables `outinsertvars` tells us where in the result we shall insert -1 where -1 means "create a new output (e.g. a neuron)
    # Thus, the result will consist of all selected indices with possibly interlaced -1s

    # TODO: Edge case when vertices are not all in graph (i.e there is at least one vertex for which its inputs or outputs are not part of vertices) and valuefun returns >= 0 might result in size inconsistency.
    # Either make sure values are > 0 for such vertices or add constraints to them!
    # Afaik this happens when only v's inputs are touched. Cleanest fix might be to separate "touch output" from "touch input". See https://github.com/DrChainsaw/NaiveNASlib.jl/issues/39
    outselectvars = Dict(v => @variable(model, [1:nout(v)], Bin) for v in vertices)
    outinsertvars = Dict(v => @variable(model, [1:nout(v)], Int, lower_bound=0) for v in vertices)

    objexpr = objective!(NeuronIndices(), s, vertices, (model=model, outselectvars=outselectvars, outinsertvars=outinsertvars, valuefun=valuefun))
    for v in vertices
        data = (model=model, outselectvars=outselectvars, outinsertvars=outinsertvars, objexpr = objexpr, valuefun = valuefun)
        vertexconstraints!(NeuronIndices(), v, s, data)
        objexpr = selectobjective!(NeuronIndices(), s, v, data)
    end

    @objective(model, Max, objexpr)

    JuMP.optimize!(model)

    !accept(s, model) && return solve_outputs_selection(fallback(s), vertices, valuefun)

    return true, extract_ininds_and_outinds(s, outselectvars, outinsertvars)...

end

accept(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, model::JuMP.Model) = JuMP.termination_status(model) != MOI.INFEASIBLE && JuMP.primal_status(model) == MOI.FEASIBLE_POINT # Beware: primal_status seems unreliable for Cbc. See MathOptInterface issue #822

selectmodel(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, v, values) = JuMP.Model(JuMP.optimizer_with_attributes(Cbc.Optimizer, "loglevel"=>0))

# First dispatch on traits to sort out things like immutable vertices
vertexconstraints!(case, v::AbstractVertex, s::AbstractJuMPΔSizeStrategy, data) = vertexconstraints!(case, trait(v), v, s, data)
vertexconstraints!(case, t::DecoratingTrait, v, s::AbstractJuMPΔSizeStrategy, data) = vertexconstraints!(case, base(t), v, s, data)
vertexconstraints!(case, t::MutationSizeTrait, v, s::AbstractJuMPΔSizeStrategy, data) = vertexconstraints!(case, s, t, v, data) # Now dispatch on strategy (and trait)
function vertexconstraints!(::NeuronIndices, ::Immutable, v, s::AbstractJuMPΔSizeStrategy, data)
     @constraint(data.model, data.outselectvars[v] .== 1)
     @constraint(data.model, data.outinsertvars[v] .== 0)
 end


function vertexconstraints!(case::NeuronIndices, s::AbstractJuMPΔSizeStrategy, t::MutationSizeTrait, v, data)
    insertconstraints!(case, s, t, v, data)
    sizeconstraint!(case, s, t, v, data)
    compconstraint!(case, s, v, (data..., vertex=v))
    inoutconstraint!(case, s, t, v, data)
end

insertconstraints!(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, ::MutationSizeTrait, v, data) = noinsertgaps!(data.model, data.outselectvars[v], data.outinsertvars[v])

"""
    noinsertgaps!(model, select, insert, maxinsert=length(outsel) * 10)

Add constraints so that `insert` does not create undefined gaps in the result of the neuron selection.

Assume `select` is a set of binary variables where `select[i] = 1` means select the output neuron at position `i` and `insert[i] = N` means insert `N` new output neurons at the position after `i`.

An example of an undefined gap is if `select = [1, 1, 0]` and `insert = [0, 0, 1]` because this results in the instruction to use existing output neurons `1 and 2` and then insert a new neuron at position `4`. 
In this example position `3` is an undefined gap as one should neither put an existing neuron there nor shall one insert new neurons. Running this method contrains `model` so that this solution is infeasible.
"""
function noinsertgaps!(model, select, insert, maxinsert=length(select) * 10)
    insert_nogap = @variable(model, [1:length(insert)], Bin)

    @constraint(model, sum(insert) <= maxinsert)

    # insert[i] == 0 if insert_nogap[i] == 1
    @constraint(model, [i=1:length(insert)], insert[i] <= 2maxinsert * insert_nogap[i])
    # Monotonicity of insert_nogap, i.e insert_nogap[i] can only be 1 if insert_nogap[i+1] is 1
    @constraint(model, [i=2:length(insert)], insert_nogap[i] <= insert_nogap[i-1])
    # Force insert_nogap to have at least as many ones as the number of not selected neurons
    @constraint(model, length(insert) - sum(select) <= sum(insert_nogap))
end

sizeconstraint!(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, t, v, data) = @constraint(data.model, sum(data.outselectvars[v]) + sum(data.outinsertvars[v])  >= 1)

function sizeconstraint!(case::NeuronIndices, s::ΔNout{Exact}, t, v, data)
    if s.vertex === v
        @constraint(data.model, sum(data.outselectvars[v]) +sum(data.outinsertvars[v]) == nout(v) + s.Δ)
    else
        sizeconstraint!(case, DefaultJuMPΔSizeStrategy(), t, v, data)
    end
end

function inoutconstraint!(::NeuronIndices, s, ::SizeAbsorb, v, data) end
function inoutconstraint!(case::NeuronIndices, s, t::SizeTransparent, v, data)
    inoutconstraint!(case, s, t, v, data.model, data.outselectvars)
    inoutconstraint!(case, s, t, v, data.model, data.outinsertvars)
end

function inoutconstraint!(::NeuronIndices, s, ::SizeStack, v, model, vardict::Dict)
    offs = 1
    var = vardict[v]
    for (i, vi) in enumerate(inputs(v))
        var_i = vardict[vi]
        # Sizes mismatch when vertex/edge was removed (or edge added)
        if nout_org(vi) == nin_org(v)[i] && offs+length(var_i)-1 <= length(var)
            @constraint(model, var_i .== var[offs:offs+length(var_i)-1])
        end
        offs += length(var_i)
    end
end

function inoutconstraint!(::NeuronIndices, s, ::SizeInvariant, v, model, vardict::Dict)
    var = vardict[v]
    for (i, vi) in enumerate(inputs(v))
        # Sizes mismatch when vertex/edge was removed (or edge added)
        nout_org(vi) == nout_org(v) || continue
        var_i = vardict[vi]
        @constraint(model, var_i .== var)
    end
end

function objective!(::NeuronIndices, s::AbstractJuMPΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex}, data) 
    scale = map(v -> max(0, maximum(data.valuefun(v))), vertices)
    noutvars = map(v -> @expression(data.model, sum(data.outselectvars[v]) + sum(data.outinsertvars[v])), vertices)
    sizetargets = map(v -> nout(v) - count(<(0), data.valuefun(v)), vertices)
    # L1 norm prevents change in vertices which does not need to change.
    # Max norm tries to spread out the change so no single vertex takes most of the change.
    return norm!(SumNorm(-0.1 => L1NormLinear(), -0.8 => MaxNormLinear()), data.model, @expression(data.model, objective[i=1:length(noutvars)], scale[i] * (noutvars[i] - sizetargets[i])), sizetargets)
end


function selectobjective!(case::NeuronIndices, s::AbstractJuMPΔSizeStrategy, v, data)
    value = valueobjective!(case, s, v, data)
    insertlast = insertlastobjective!(case, s, v, data)
    return @expression(data.model, data.objexpr + value + insertlast)
end

valueobjective!(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, v, data) = @expression(data.model, sum(data.valuefun(v) .* data.outselectvars[v]))

function insertlastobjective!(::NeuronIndices, s, v, data)
    insvars = data.outinsertvars[v]
    preferend = collect(length(insvars) : -1 : 1)
    scale = minimum(x -> x > 0 ? x : typemax(x), data.valuefun(v)) / sum(preferend) 
    return @expression(data.model, -0.1scale*sum(insvars .* preferend))
end

function extract_ininds_and_outinds(s, outselectvars::Dict, outinsertvars::Dict)
    outinds = Dict([v => extract_inds(s, outselectvars[v], outinsertvars[v]) for v in keys(outselectvars)])
    ininds = Dict([v => getall(outinds, inputs(v)) for v in keys(outselectvars)])
    return ininds, outinds
end

function extract_inds(::AbstractJuMPΔSizeStrategy, selectvars::T, insertvars::T) where T <: AbstractVector{JuMP.VariableRef}
    # insertvar is N at indices where a N new output neurons shall be added
    insert = round.(Int, JuMP.value.(insertvars))
    selected = findall(xi -> xi > 0, JuMP.value.(selectvars))
    return join_extracted_inds(selected, insert)
end

function join_extracted_inds(selected, insert)
    result = Vector{Int}(undef, length(selected) + sum(insert))

    # Expected result of algorithm: initialize result = selected, then insert insert[i] -1 elements after each index i in result  

    i = 1 # result index
    j = 1 # selected index
    k = 1 # insert index
    while i <= length(result)
        if j <= i && j <= length(selected)
            result[i] = selected[j]
            j += 1
            i += 1
        end
        if insert[k] > 0
            start = i
            stop = i+insert[k]-1
            result[start:stop] .= -1
            i += insert[k]
        end
        k += 1
    end
    return result
end


