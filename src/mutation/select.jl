
"""
    NeuronIndices

Treat vertices as having parameters which represents neurons when formulating the size change problem.

This means that individual indices will be aligned so that the function to the largest extent possible is the same after resizing.
"""
struct NeuronIndices end

Δsizetype(vs::AbstractVector{<:AbstractVertex}) = partialsort!(Δsizetype.(vs), 1; by=Δsizetypeprio, rev=true)
Δsizetype(v::AbstractVertex, seen=Set()) = v in seen ? nothing : Δsizetype(trait(v), v, push!(seen, v))
Δsizetype(t::DecoratingTrait, v, seen) = Δsizetype(base(t), v, seen)
Δsizetype(t::MutationTrait, v, seen) = Δsizetype(t, base(v), seen)
Δsizetype(::MutationTrait, ::InputVertex, seen) = nothing
Δsizetype(::MutationTrait, v::CompVertex, seen) = Δsizetype(v.computation)
Δsizetype(f) = NeuronIndices()

# TODO: Label elemwise and concatenation to allow them to return some low prio Δsizetype? Now things like BatchNorm will not have an opinion on its Δsizetype
# Perhaps better idea: pass down trait to Δsizetype(f) and default to low prio Δsizetype for SizeTransparent? Requires that one always looks at all vertices, but that seems to be the requirement anyways
Δsizetype(::SizeTransparent, v, seen) = partialsort!(vcat(map(vi -> Δsizetype(vi, seen), inputs(v)), map(vo -> Δsizetype(vo, seen), outputs(v))), 1; by=Δsizetypeprio, rev=true)

Δsizetypeprio(::Nothing) = 0
Δsizetypeprio(::ScalarSize) = 1
Δsizetypeprio(::NeuronIndices) = 2

"""
    defaultutility(v::AbstractVertex) 

Default function used to calculate utility of output neurons.

Implement either `defaultutility(f)` or `defaultutility(t, f)` where `f` is the computation performed 
by `CompVertex` and `t` is [`trait(v)`](@ref) to set the default for `f`. 

# Examples

```jldoctest
julia> using NaiveNASlib, Statistics

julia> struct Affine{T}
        W::Matrix{T}
       end;

julia> NaiveNASlib.defaultutility(l::Affine) = mean(abs, l.W; dims=2);

julia> NaiveNASlib.defaultutility(Affine(ones(2,3)))
2×1 Matrix{Float64}:
 1.0
 1.0
```
"""
defaultutility(v::AbstractVertex) = defaultutility(trait(v), v)
defaultutility(t, v::AbstractVertex) = defaultutility(t, base(v))
defaultutility(t, ::InputVertex) = 1
defaultutility(t, v::CompVertex) = defaultutility(t, v.computation)
defaultutility(t, f) = resolve_utility(t, defaultutility(f))

struct NoDefaultUtility end

defaultutility(f) = NoDefaultUtility()

resolve_utility(t::DecoratingTrait, u::NoDefaultUtility) = resolve_utility(base(t),u)
resolve_utility(::MutationTrait, ::NoDefaultUtility) = 1
resolve_utility(::SizeTransparent, ::NoDefaultUtility) = 0 # rationale is that utility of other vertices will be tied to this anyways
resolve_utility(t, val) = val

Δsize!(::Nothing, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}) = false

# Main entry point for computing and applying size changes with NeuronIndices

Δsize!(case::NeuronIndices, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}) = Δsize!(defaultutility, case,s , vs)
function Δsize!(utilityfun, case::NeuronIndices, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex})
    success, ins, outs = solve_outputs_selection(s, vs, utilityfun)
    if success
        applyΔsize!(case, s, ins, outs, vs)
    end
    return success
end

function Δsize!(utilityfun, case::NeuronIndices, s::TruncateInIndsToValid, vs::AbstractVector{<:AbstractVertex})
    success, ins, outs = solve_outputs_selection(s.strategy, vs, utilityfun)
    if success
        for (vv, ininds) in ins
            for innr in eachindex(ininds)
                ininds[innr] = aligntomax(nin(vv)[innr], ininds[innr])
            end
            newins, newouts = align_outs_to_ins(vv, ins[vv], outs[vv])
            ins[vv] = newins
            outs[vv] = newouts
        end
        applyΔsize!(case, s, ins, outs, vs)
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
function align_outs_to_ins(::SizeInvariant, v, ins::AbstractVector, outs)
    isempty(ins) && return ins, outs
    inds = ins[1]
    newins = repeat([inds], length(ins))
    newouts = ismissing(outs) ? outs : inds
    return newins, newouts
end

# Main entry point for applying size changes with NeuronIndices after we have compted them
# Peel off all DecoratingJuMPΔSizeStrategies until we find either an AbstractAfterΔSizeStrategy (which is needed in after_Δnin etc)
# or we hit the bottom (in the form of an AbstractΔSizeStrategy). 
function applyΔsize!(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, ins::AbstractDict, outs::AbstractDict, vs::AbstractVector{<:AbstractVertex}) 
    applyΔsize!(case, base(s), ins, outs, vs)
end
function applyΔsize!(case::NeuronIndices, s::AbstractAfterΔSizeStrategy, ins::AbstractDict, outs::AbstractDict, vs::AbstractVector{<:AbstractVertex}) 
    _applyΔsize!(case, s, ins, outs, vs)
end
function applyΔsize!(case::NeuronIndices, s::AbstractΔSizeStrategy, ins::AbstractDict, outs::AbstractDict, vs::AbstractVector{<:AbstractVertex}) 
    _applyΔsize!(case, s, ins, outs, vs)
end

function _applyΔsize!(case::NeuronIndices, s, ins::AbstractDict, outs::AbstractDict, vs::AbstractVector{<:AbstractVertex})

    Δnins = map(vs) do vi
        vins = ins[vi]
        ismissing(vins) && return false
        mapreduce(|, zip(vins, nin(vi)); init=false) do (vin, ninorg)
            ismissing(vin) && return false
            return length(vin) != ninorg
        end
    end
    Δnouts = [length(outs[vi]) != nout(vi) for vi in vs]

    for vi in vs
        applyΔsize!(case, s, vi, ins[vi], outs[vi])
    end

    for (i, vi) in enumerate(vs)
        after_Δnin(s, vi, ins[vi], Δnins[i])
        after_Δnout(s, vi, outs[vi], Δnouts[i])
    end
end

# Main entry point for applying size changes to a single vertex with NeuronIndices

function applyΔsize!(case::NeuronIndices, s::WithKwargs, v::AbstractVertex, ins::AbstractVector, outs::AbstractVector; kws...) 
    applyΔsize!(case, base(s), v, ins, outs; kws..., s.kwargs...)
end
function applyΔsize!(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, v::AbstractVertex, ins::AbstractVector, outs::AbstractVector; kws...)
     applyΔsize!(case, base(s), v, ins, outs; kws...)
end
function applyΔsize!(case::NeuronIndices, s::AbstractΔSizeStrategy, v::AbstractVertex, ins::AbstractVector, outs::AbstractVector; kws...) 
    applyΔsize!(case, v, ins, outs; kws...)
end

function applyΔsize!(::NeuronIndices, v::InputSizeVertex, ins::AbstractVector, outs::AbstractVector; kwargs...) 
    if !all(isempty, skipmissing(ins))
        throw(ArgumentError("Try to change input neurons of InputVertex $(name(v)) to $ins"))
    end 
    if outs != 1:nout(v)
        throw(ArgumentError("Try to change output neurons of InputVertex $(name(v)) to $outs"))
    end
end

function applyΔsize!(case::NeuronIndices, v::AbstractVertex, ins::AbstractVector, outs::AbstractVector; kwargs...) 
    applyΔsize!(case, base(v), ins, outs; kwargs...)
end
function applyΔsize!(case::NeuronIndices, v::CompVertex, ins::AbstractVector, outs::AbstractVector; kwargs...) 
    # I don't love how we go back to Δsize! from here, but I don't want to export applyΔsize! and I can't think of another name right now...
    Δsize!(case, v.computation, ins, outs; kwargs...)
end
Δsize!(::NeuronIndices, f, ins::AbstractVector, outs::AbstractVector; kwargs...) = Δsize!(f, ins, outs; kwargs...)
"""
    Δsize!(f::F, ins::AbstractVector, outs::AbstractVector; kwargs...) 

Apply the changes to `f` so that input neurons in `ins` and output neurons in `outs` are selected and/or inserted.

Argument `outs` is a vector of indices to select/insert while `ins` has one vector of indices per input vertex.

Shall be implemented for any type `F` which holds parameters for which the shape shall be modified by NaiveNASlib.

Tip: the function [`parselect`](@ref) can be used to change parameter arrays according to `ins` and `outs`.

Tip: `kwargs` can be passed using [`WithKwargs`](@ref).
"""
function Δsize!(f, ins::AbstractVector, outs::AbstractVector; kwargs...) end

"""
    parselect(pars::AbstractArray{T,N}, elements_per_dim...; newfun = (T, dim, size...) -> 0) where {T, N}

Return a new array of same type as `pars` which has a subset of the elements of `pars` as well as potentially
new elements.

Which elements to select/insert is determined by `elements_per_dim` which is a `Pair{Int, Vector{Int}}` mapping 
dimension (first memeber) to which elements to select/insert in that dimension (second memeber).

For a each `dim=>elems` pair, the following holds: `selectdim(output, dim, i) == selectdim(pars, dim, elems[i])` 
if `elems[i]` is positive and `selectdim(output, dim, i) .== newfun(T, dim, size)[j]` if `elems[i]` is the `j:th`
negative value and `size` is `sum(elems .< 0)`.

# Examples
```jldoctest
julia> using NaiveNASlib, NaiveNASlib.Extend

julia> pars = reshape(1:3*5, 3,5)
3×5 reshape(::UnitRange{Int64}, 3, 5) with eltype Int64:
 1  4  7  10  13
 2  5  8  11  14
 3  6  9  12  15
 
julia> NaiveNASlib.parselect(pars, 1 => [-1, 1,3,-1,2], 2=>[3, -1, 2, 1]; newfun = (T, d, s...) -> fill(-T(d), s))
5×4 Matrix{Int64}:
 -1  -2  -1  -1
  7  -2   4   1
  9  -2   6   3
 -1  -2  -1  -1
  8  -2   5   2
```
"""
function parselect(pars::AbstractArray{T,N}, elements_per_dim...; newfun = (T, dim, size...) -> 0) where {T, N}
    psize = collect(size(pars))
    assign = repeat(Any[Colon()], N)
    access = repeat(Any[Colon()], N)

    elements_per_dim = filter(!ismissing ∘ last, elements_per_dim)

    for (dim, elements) in elements_per_dim
        psize[dim] = length(elements)
    end
    newpars = similar(pars, psize...)

    for (dim, elements) in elements_per_dim
        indskeep = filter(ind -> ind > 0, elements)
        indsmap = elements .> 0
        newmap = .!indsmap

        assign[dim] = findall(indsmap)
        access[dim] = indskeep
        tsize = copy(psize)
        tsize[dim] = sum(newmap)
        selectdim(newpars, dim, newmap) .= newfun(T, dim, tsize...)
    end

    newpars[assign...] = pars[access...]
    return newpars
end
parselect(::Missing, args...;kwargs...) = missing


function solve_outputs_selection(s::LogΔSizeExec, vertices::AbstractVector{<:AbstractVertex}, utilityfun)
    @logmsg s.level s.msgfun(vertices[1])
    return solve_outputs_selection(base(s), vertices, utilityfun)
end

solve_outputs_selection(::ΔSizeFailNoOp, vs::AbstractVector{<:AbstractVertex}, ::Any) = false, Dict(v => [1:n for n in nin(v)] for v in vs), Dict(v => 1:nout(v) for v in vs)
solve_outputs_selection(s::ThrowΔSizeFailError, vertices::AbstractVector{<:AbstractVertex}, ::Any) = throw(ΔSizeFailError(s.msgfun(vertices)))
solve_outputs_selection(s::WithUtilityFun, vs::AbstractVector{<:AbstractVertex}, ::Any) = solve_outputs_selection(s.strategy, vs, s.utilityfun) 
"""
    solve_outputs_selection(s::AbstractΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex}, utilityfun)

Returns a tuple `(success, nindict, noutdict)` where `nindict[vi]` are new input neuron indices and `noutdict[vi]` are new output neuron indices for each vertex `vi` in `vertices`.

The function generally tries to maximize `sum(utilityfun(vi) .* selected[vi]) ∀ vi in vertices` where `selected[vi]` is all elements in `noutdict[vi]` larger than 0 (negative values 
in `noutdict` indicates a new output shall be inserted at that position). This however is up to the implementation of the `AbstractΔSizeStrategy s`.

Since selection of outputs is not guaranteed to work in all cases, a flag `success` is also returned. If `success` is `false` then applying the new indices may (and probably will) fail.
"""
function solve_outputs_selection(s::AbstractJuMPΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}, orgutilityfun)
    utilityfun = remaputilityfun(orgutilityfun, s, vs)
    model = selectmodel(NeuronIndices(), s, vs, utilityfun)
    case = NeuronIndices()

    # TODO: Cache and rescale utilityfun?

    # The binary variables `outselectvars` tells us which existing output indices to select
    # The integer variables `outinsertvars` tells us where in the result we shall insert -1 where -1 means "create a new output (e.g. a neuron)
    # Thus, the result will consist of all selected indices with possibly interlaced -1s
    outselectvars, outinsertvars, noutdict = createselectvars!(case, s, vs, (;model, utilityfun))

    objexpr = sizeobjective!(case, s, vs, (;model, outselectvars, outinsertvars, noutdict, utilityfun))
    for v in vs
        data = (;model, outselectvars, outinsertvars, noutdict, objexpr, utilityfun)
        vertexconstraints!(case, v, s, data)
        objexpr = selectobjective!(case, s, v, data)
    end

    @objective(model, Max, objexpr)

    JuMP.optimize!(model)

    !accept(case, s, model) && return   let fbstrat = fallback(s)
                                            solve_outputs_selection(fbstrat, add_participants!(fbstrat, copy(vs)), orgutilityfun)
                                        end

    return true, extract_ininds_and_outinds(s, outselectvars, outinsertvars)...
end

remaputilityfun(utilityfun, s::DecoratingJuMPΔSizeStrategy, vs, args...) = remaputilityfun(utilityfun, base(s), vs, args...)
function remaputilityfun(utilityfun, ::AbstractJuMPΔSizeStrategy, vs, floorval=1e-2, ceilval=1e5)
    cache = Dict{eltype(vs), Any}(v => utilityfun(v) for v in vs)

    # For any problem which one thinks can be solved by writing code one will always generate new problems
    # Learn to keep the number of moving parts in a codebase to a minimum or spend an eternity writing 
    # code like this
    !any(vs) do v
        issizemutable(v) || return false
        any(>(0), cache[v])
    end && return v -> cache[v]

    # We would prefer the range of the utility values to stay well within typical solver tolerances
    # Not sure how to do this correctly. Some papers point towards scaling the cost/constraint matrix, 
    # but it does not seem to be availble from JuMP

    # What this ad-hoc-y algorithm tries to do is to check if the magnitudes are within bounds [floorval, ceilval]
    # If they are not, it tries to adjust them to be so by uniform scaling.
    # The scaling is limited so that it never pushes something out of bounds.
    # If magnitudes outside both bounds exist (i.e both too large and too small values exist) it does not do anything. 
    # TODO: Use quantiles to not let outliers prevent scaling? Some optimization criterion for maximizing the fraction of values within bounds? 

    # Don't let 0s affect the result or else all those SizeTransparent vertices is going to push us to max scaling all the time.
    minval, maxval = extrema(abs, Iterators.filter(!=(0), Iterators.flatten(val for (v, val) in cache if issizemutable(v))))
    lb = max(minval, floorval)
    ub = min(maxval, ceilval)
    maxscaledown = lb / floorval
    scaledown = min(maxscaledown, maxval / ub)
    maxscaleup = ceilval / ub
    scaleup = min(maxscaleup, lb / minval)

    scal = scaledown > scaleup ? 1 / scaledown : scaleup

    for (v, val) in cache
        if issizemutable(v)
            cache[v] = val .* scal
        end
    end
    return v -> cache[v]
end

createselectvars!(case, s::DecoratingJuMPΔSizeStrategy, vs, data) = createselectvars!(case ,base(s), vs, data)
function createselectvars!(case, ::AbstractJuMPΔSizeStrategy, vs, data)  
    model = data.model
    utilityfun = data.utilityfun
    # We use the lenght of utilityfun as the number of outputs to select as we will use it to add utility to each variable,
    # so their lenghts must not mismatch
    # They might mismatch when there are parameters attached to some size transparent vertex (e.g BatchNorm or even
    # concatenation with some computed activation utility metric) and we are in the process of aligning sizes after 
    # adding/removing an edge.
    # Forcing the user to provide a length of utilityfun matching nout(v) in this case is not really fair as it is we who 
    # caused the mismatch.
    # TODO: The indicies alignment below will not be ideal in this case. Perhaps it can be mitigated with some strategy 
    # providing information on how the edges have changed so that we can better align what hasn't changed.
    selectsizes = map(vs) do v
        outvals = utilityfun(v)
        return outvals isa Number ? nout(v) : length(outvals)
    end

    outselectvars = Dict(v => @variable(model, [1:nvals], Bin) for (v,nvals) in zip(vs, selectsizes))
    outinsertvars = Dict(v => @variable(model, [1:nout(v)], Int, lower_bound=0) for v in vs)

    # This is the same variable name as used when adjusting size only. It allows us to delegate alot of size changing strategies to ScalarSize. 
    # Drawback is that it will yield a fair bit of back and forth so maybe it is too clever for its own good
    noutdict = Dict(v => @expression(model, sum(outselectvars[v]) + sum(outinsertvars[v])) for v in vs)

    return outselectvars, outinsertvars, noutdict
end

selectmodel(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, vs, utilityfun) = selectmodel(case, base(s), vs, utilityfun) 
selectmodel(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, vs, utilityfun) = JuMP.Model(JuMP.optimizer_with_attributes(Cbc.Optimizer, "loglevel"=>0, "seconds" => 20))
function selectmodel(case::NeuronIndices, s::TimeLimitΔSizeStrategy, vs, utilityfun) 
    model = selectmodel(case, base(s), vs, utilityfun)
    JuMP.set_time_limit_sec(model, s.limit)
    return model
end

function vertexconstraints!(case::NeuronIndices, ::Immutable, v, s::AbstractJuMPΔSizeStrategy, data)
     JuMP.set_lower_bound.(data.outselectvars[v], 1)
     JuMP.set_upper_bound.(data.outinsertvars[v], 0)
     for (vi, outsize) in zip(inputs(v), nin(v))
        @constraint(data.model, data.noutdict[vi] == outsize)
     end
     # This is needed e.g. to apply (impossible) size constraints incase someone wants to change the input size of an output vertex
     vertexconstraints!(case ,s, v, data)
end

# ScalarSize constraints happen to be good here, but we must ensure that vertexconstraints!(::NeuronIndices,...) gets called instead of vertexconstraints!(::ScalarSize,...)
# For AlignNinToNout we could just have broken out the call to vertexconstraints!, but for AlignNinToNoutVertices we need to call it in the middle of the function
vertexconstraints!(::NeuronIndices, v::AbstractVertex, s::AlignNinToNout, data) = vertexconstraints!(ScalarSize(), v, s, data, NeuronIndices())
vertexconstraints!(::NeuronIndices, v::AbstractVertex, s::AlignNinToNoutVertices, data) = vertexconstraints!(ScalarSize(), v::AbstractVertex, s::AlignNinToNoutVertices, data, NeuronIndices())

function vertexconstraints!(case::NeuronIndices, s::AbstractJuMPΔSizeStrategy, v, data)
    insertconstraints!(case, s, v, data)
    sizeconstraint!(case, s, v, data)
    compconstraint!(case, s, v, (data..., vertex=v))
    inoutconstraint!(case, s, v, data)
end

insertconstraints!(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, v, data) = insertconstraints!(case, base(s), v, data) 
insertconstraints!(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, v, data) = noinsertgaps!(data.model, data.outselectvars[v], data.outinsertvars[v])

"""
    noinsertgaps!(model, select, insert, maxinsert=length(outsel) * 10)

Add constraints so that `insert` does not create undefined gaps in the result of the neuron selection.

Assume `select` is a set of binary variables where `select[i] = 1` means select the output neuron at position `i` and `insert[i] = N` means insert `N` new output neurons at the position after `i`.

An example of an undefined gap is if `select = [1, 1, 0]` and `insert = [0, 0, 1]` because this results in the instruction to use existing output neurons `1 and 2` and then insert a new neuron at position `4`. 
In this example position `3` is an undefined gap as one should neither put an existing neuron there nor shall one insert new neurons. Running this method constrains `model` so that this solution is infeasible.
"""
function noinsertgaps!(model, select, insert, maxinsert=max(length(select) * 10, 200)) # TODO: get maxinsert from strategy instead?
    # See  https://discourse.julialang.org/t/help-with-constraints-to-select-and-or-insert-columns-to-a-matrix/63654
    insert_nogap = @variable(model, [1:length(insert)], Bin)

    @constraint(model, sum(insert) <= maxinsert)
    JuMP.set_upper_bound.(filter(!JuMP.has_upper_bound, insert), maxinsert)

    # insert[i] == 0 if insert_nogap[i] == 1
    @constraint(model, [i=1:length(insert)], insert[i] <= 2maxinsert * insert_nogap[i])
    # Monotonicity of insert_nogap, i.e insert_nogap[i] can only be 1 if insert_nogap[i+1] is 1
    @constraint(model, [i=2:length(insert)], insert_nogap[i] <= insert_nogap[i-1])
    # Force insert_nogap to have at least as many ones as the number of not selected neurons
    @constraint(model, length(insert) - sum(select) <= sum(insert_nogap))
end

sizeconstraint!(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, v, data) = sizeconstraint!(case, base(s), v, data)
sizeconstraint!(::NeuronIndices, s::AbstractJuMPΔSizeStrategy, v, data) = sizeconstraint!(ScalarSize(), DefaultJuMPΔSizeStrategy(), v, data)
sizeconstraint!(case::NeuronIndices, s::ΔNoutMix, v, data) = sizeconstraint!(case, s.exact, v, data)
sizeconstraint!(::NeuronIndices, s::ΔNout{Exact}, v, data) = sizeconstraint!(ScalarSize(), s, v, data)

inoutconstraint!(case, s::DecoratingJuMPΔSizeStrategy, v, data) = inoutconstraint!(case, base(s), v, data) 
inoutconstraint!(case, s, v, data) = inoutconstraint!(case, s, trait(v), v, data) 
inoutconstraint!(case::NeuronIndices, s, t::DecoratingTrait, v, data) = inoutconstraint!(case, s, base(t), v, data) 


function inoutconstraint!(::NeuronIndices, s, ::MutationTrait, v, data) end
function inoutconstraint!(case::NeuronIndices, s, t::SizeTransparent, v, data)
    onemismatch  = inoutconstraint!(case, s, t, v, data.model, data.outselectvars)
    onemismatch |= inoutconstraint!(case, s, t, v, data.model, data.outinsertvars)
    
    # We need to ensure that total sizes align if there was a mismatch
    if onemismatch
        ninconstraint!(case, s, v, data)
    end
end

function inoutconstraint!(::NeuronIndices, s, t::SizeStack, v, model, vardict::Dict)
    offs = 1
    var = vardict[v]
    onemismatch = false
    for (i, vi) in enumerate(inputs(v))
        var_i = vardict[vi]
        # Sizes mismatch when vertex/edge was removed (or edge added)
        if length(var_i) == nin(v)[i] && offs+length(var_i)-1 <= length(var)
            @constraint(model, var_i .== var[offs:offs+length(var_i)-1])
        else
            onemismatch = true
        end
        offs += length(var_i)
    end
    # Length var != nout means that we need to enforce the SizeStack constraint that sum(nin.(inputs(v))) == nout(v)
    # as this indicates that there is some parameter array which needs to be aligned with the new size
    return onemismatch || length(var) != nout(v)
end

function inoutconstraint!(::NeuronIndices, s, ::SizeInvariant, v, model, vardict::Dict)
    var = vardict[v]
    onemismatch = false
    for vi in inputs(v)
        # Sizes mismatch when vertex/edge was removed (or edge added)
        var_i = vardict[vi]
        if length(var_i) == length(var)
            @constraint(model, var_i .== var)
        else
            onemismatch = true
        end
    end
    return onemismatch
end

# Minus sign because we maximize objective function in case NeuronIndices
sizeobjective!(case::NeuronIndices, s::AbstractJuMPΔSizeStrategy, vertices, data) = -objective!(case, s, vertices, data)

objective!(case::NeuronIndices, s::ΔNoutMix, vertices, data) = objective!(case, s.relax, vertices, data)
objective!(case::NeuronIndices, s::ΔNout{Relaxed}, vertices, data) = noutrelax!(case, s.Δs, vertices, data)

objective!(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, vertices, data) = objective!(case, base(s), vertices, data) 
function objective!(::NeuronIndices, s::AbstractJuMPΔSizeStrategy, vertices, data) 
    # Or else the problem becomes infeasible
    isempty(vertices) && return @expression(data.model, 0)
    
    scale = map(v -> maximum(abs, data.utilityfun(v)), vertices)
    noutvars = map(v -> data.noutdict[v], vertices)
    sizetargets = map(v -> sizetarget(data.utilityfun, v), vertices)
    # L1 norm prevents change in vertices which does not need to change.
    # Max norm tries to spread out the change so no single vertex takes most of the change.
    return norm!(SumNorm(0.1 => L1NormLinear(), 0.8 => MaxNormLinear()), data.model, @expression(data.model, objective[i=1:length(noutvars)], scale[i] * (noutvars[i] - sizetargets[i])), sizetargets)
end

function sizetarget(utilityfun, v)
    util = utilityfun(v)
    # Negative scalar utility is probably never useful, but lets make it consistent with an all negative array
    util isa Number && return util < 0 ? 1 : nout(v)
    return max(1, nout(v) - count(<(0), util))
end

function selectobjective!(case::NeuronIndices, s::AbstractJuMPΔSizeStrategy, v, data)
    utility = utilityobjective!(case, s, v, data)
    @objective(data.model, Max, utility)
    # Also makes us prefer to not insert
    insertlast = insertlastobjective!(case, s, v, data)
    return @expression(data.model, data.objexpr + utility + insertlast)
end
utilityobjective!(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, v, data) = utilityobjective!(case, base(s), v, data)
utilityobjective!(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, v, data) = @expression(data.model, sum(data.utilityfun(v) .* data.outselectvars[v]))

insertlastobjective!(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, v, data) = insertlastobjective!(case::NeuronIndices, base(s), v, data)
function insertlastobjective!(::NeuronIndices, s, v, data)
    insvars = data.outinsertvars[v]
    
    isempty(insvars) && return @expression(data.model, 0)

    preferend = collect(length(insvars) : -1 : 1)
    scale = minimum(abs, data.utilityfun(v)) / sum(preferend) 
    return @expression(data.model, -0.1scale*sum(insvars .* preferend))
end

extract_ininds_and_outinds(s::DecoratingJuMPΔSizeStrategy, outselectvars::Dict, outinsertvars::Dict) = extract_ininds_and_outinds(base(s), outselectvars, outinsertvars)
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
        if k <= length(insert) && insert[k] > 0
            start = i
            stop = i+insert[k]-1
            result[start:stop] .= -1
            i += insert[k]
        end
        k += 1
    end
    return result
end


