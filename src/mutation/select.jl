
"""
    NeuronIndices

Treat vertices as having parameters which represents neurons when formulating the size change problem.

This means that individual indices will be aligned so that the function to the largest extent possible is the same after resizing.
"""
struct NeuronIndices end

# Just another name for Δsize with the corresponding direction
Δnin!(v::AbstractVertex, Δ, Δs...) = Δsize!(Input(), v => (Δ, Δs...))
Δnout!(v::AbstractVertex, Δ) = Δsize!(Output(), v=>Δ)
Δnin!(args...) = Δsize!(Input(), args...)
Δnout!(args...) = Δsize!(Output(), args...)
Δnin!(ps::Pair{<:AbstractVertex}...) = Δsize!(Input(), ps...)
Δnout!(ps::Pair{<:AbstractVertex}...) = Δsize!(Output(), ps...)

Δnin!(valuefun, v::AbstractVertex, Δ, Δs...) = Δsize!(valuefun, Input(), v => (Δ, Δs...))
Δnout!(valuefun, v::AbstractVertex, Δ) = Δsize!(valuefun, Output(), v=>Δ)
Δnin!(valuefun, p::Pair{<:AbstractVertex}, ps::Pair...) = Δsize!(valuefun, Input(), p, ps...)
Δnout!(valuefun, p::Pair{<:AbstractVertex}, ps::Pair...) = Δsize!(valuefun, Output(), p, ps...)
Δnin!(valuefun, d::AbstractDict) = Δsize!(valuefun, Input(), d)
Δnout!(valuefun, d::AbstractDict) = Δsize!(valuefun, Output(), d)


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
    default_outvalue(v::AbstractVertex) 

Default function used to calculate value of output neurons when `NeuronIndices` is used and no function is provided.

Implement `default_outvalue(t, f)` where `f` is the computation performed by `CompVertex` to set the default for `f` and `t` is the [`trait`](@ref) of `v`. 

# Examples

```julia-repl
julia> struct Affine{T}
    W::Matrix{T}
end;

# Default is weight magnitude
julia> NaiveNASlib.default_outvalue(t, l::Affine) = mean(abs, l.W; dims=2);

```

"""
default_outvalue(v::AbstractVertex) = default_outvalue(trait(v), v)
default_outvalue(t, v::AbstractVertex) = default_outvalue(t, base(v))
default_outvalue(t, ::InputVertex) = 1
default_outvalue(t, v::CompVertex) = default_outvalue(t, v.computation)
default_outvalue(t, f) = default_outvalue(f)
default_outvalue(f) = 1

"""
    Δsize!(d::Direction, v::AbstractVertex, Δ...)

Change size of `v` by `Δ` in direction `d`.
"""
Δsize!(d::Input, v::AbstractVertex, Δs::Maybe{<:Integer}...) = Δsize!(ΔNin(v, Δs), all_in_Δsize_graph(v, d))
Δsize!(d::Output, v::AbstractVertex, Δ::Integer) = Δsize!(ΔNout(v, Δ), all_in_Δsize_graph(v, d))
Δsize!(d::Input, args...) = Δsize!(ΔNin(args...), all_in_Δsize_graph(args, d))
Δsize!(d::Output, args...) = Δsize!(ΔNout(args...), all_in_Δsize_graph(args, d))

Δsize!(f, d::Input, args...) = Δsize!(f, ΔNin(args...), all_in_Δsize_graph(args, d))
Δsize!(f, d::Output, args...) = Δsize!(f, ΔNout(args...), all_in_Δsize_graph(args, d))

Δsize!(valuefun, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}) = Δsize!(valuefun, Δsizetype(vs), s, vs)
Δsize!(s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}) = Δsize!(Δsizetype(vs), s, vs)


"""
    Δsize!(valuefun, g::CompGraph)
    Δsize!(valuefun, s::AbstractΔSizeStrategy, g::CompGraph)
    Δsize!(valuefun, v::AbstractVertex)
    Δsize!(valuefun, s::AbstractΔSizeStrategy, v::AbstractVertex)

Change output neurons of all vertices of graph `g` (or graph to which `v` is connected) according to the provided `AbstractΔSizeStrategy s` (default `OutSelect{Exact}`).

Return true of operation was successful, false otherwise.

Argument `valuefun` provides a vector `value = valuefun(vx)` for any vertex `vx` in the same graph as `v` where `value[i] > value[j]` indicates that output neuron index `i` shall be preferred over `j` for vertex `vx`.

If provided, `Direction d` will narrow down the set of vertices to evaluate so that only vertices which may change as a result of changing size of `v` are considered.
"""
Δsize!(g::CompGraph) = Δsize!(default_outvalue, g::CompGraph)
Δsize!(valuefun, g::CompGraph) = Δsize!(valuefun, DefaultJuMPΔSizeStrategy(), g)

Δsize!(s::AbstractΔSizeStrategy, g::CompGraph) = Δsize!(default_outvalue, s::AbstractΔSizeStrategy, g::CompGraph)
Δsize!(valuefun, s::AbstractΔSizeStrategy, g::CompGraph) = Δsize!(valuefun, s, vertices(g))

Δsize!(v::AbstractVertex) = Δsize!(default_outvalue, v::AbstractVertex)
Δsize!(valuefun, v::AbstractVertex) = Δsize!(valuefun, DefaultJuMPΔSizeStrategy(), v)

Δsize!(s::AbstractΔSizeStrategy, v::AbstractVertex) =Δsize!(default_outvalue, s::AbstractΔSizeStrategy, v::AbstractVertex) 
Δsize!(valuefun, s::AbstractΔSizeStrategy, v::AbstractVertex) = Δsize!(valuefun, s, all_in_graph(v))
function Δsize!(valuefun, s::SelectDirection, v::AbstractVertex)
    d = Δdirection(s.strategy)
    d === nothing && return true
    return Δsize!(valuefun, s.strategy, d, v)
end

Δdirection(::AbstractΔSizeStrategy) = Both()
Δdirection(s::LogΔSizeExec) = Δdirection(s.andthen)
Δdirection(::ΔNout) = Output()

Δsize!(::Nothing, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}) = false

Δsize!(case::NeuronIndices, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}) = Δsize!(default_outvalue, case,s , vs)
function Δsize!(valuefun, case::NeuronIndices, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex})
    success, ins, outs = solve_outputs_selection(s, vs, valuefun)
    if success
        Δsize!(case, ins, outs, vs)
    end
    return success
end

function Δsize!(valuefun, case::NeuronIndices, s::TruncateInIndsToValid, vs::AbstractVector{<:AbstractVertex})
    success, ins, outs = solve_outputs_selection(s.strategy, vs, valuefun)
    if success
        for (vv, ininds) in ins
            for innr in eachindex(ininds)
                ininds[innr] = aligntomax(nin(vv)[innr], ininds[innr])
            end
            newins, newouts = align_outs_to_ins(vv, ins[vv], outs[vv])
            ins[vv] = newins
            outs[vv] = newouts
        end
        Δsize!(case, ins, outs, vs)
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

"""
    Δsize!(case::NeuronIndices, ins::Dict, ins::AbstractDict, outs::AbstractDict, vertices::AbstractVector{<:AbstractVertex})

Set input and output indices of each `vi` in `vs` to `outs[vi]` and `ins[vi]` respectively.
"""
function Δsize!(case::NeuronIndices, ins::AbstractDict, outs::AbstractDict, vs::AbstractVector{<:AbstractVertex})

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
        Δsize!(case, OnlyFor(), vi, ins[vi], outs[vi])
    end

    for (i, vi) in enumerate(vs)
        after_Δnin(vi, ins[vi], Δnins[i])
        after_Δnout(vi, outs[vi], Δnouts[i])
    end
end

Δsize!(case::NeuronIndices, s::OnlyFor, v::AbstractVertex, ins::AbstractVector, outs::AbstractVector) = Δsize!(case, s, base(v), ins, outs)
Δsize!(::NeuronIndices, ::OnlyFor, v::CompVertex, ins::AbstractVector, outs::AbstractVector) = Δsize!(v.computation, ins, outs)
function Δsize!(f, ins::AbstractVector, outs::AbstractVector) end

function Δsize!(::NeuronIndices, ::OnlyFor, v::InputSizeVertex, ins::AbstractVector, outs::AbstractVector) 
    if !all(isempty, skipmissing(ins))
        throw(ArgumentError("Try to change input neurons of InputVertex $(name(v)) to $ins"))
    end 
    if outs != 1:nout(v)
        throw(ArgumentError("Try to change output neurons of InputVertex $(name(v)) to $outs"))
    end
end

"""
    parselect(pars::AbstractArray{T,N}, elements_per_dim...; newfun = zeros) where {T, N}

Return a new `AbstractArray{T, N}` which has a subset of the elements of `pars`.

Which elements to select is determined by `elements_per_dim` which is a `Pair{Int, Vector{Int}}` mapping dimension (first memeber) to which elements to select in that dimension (second memeber).

For a single `dim=>elems` pair, the following holds: `selectdim(output, dim, i) == selectdim(pars, dim, elems[i])` if `elems[i]` is positive and `selectdim(output, dim, i) .== newfun(T, dim, size)[j]` if `elems[i]` is the `j:th` negative value and `size` is `sum(elems .< 0)`.

# Examples
```julia-repl

julia> pars = reshape(1:3*5, 3,5)
3×5 reshape(::UnitRange{Int64}, 3, 5) with eltype Int64:
 1  4  7  10  13
 2  5  8  11  14
 3  6  9  12  15

 julia> NaiveNASlib.parselect(pars, 1 => [-1, 1,3,-1,2], 2=>[3, -1, 2], newfun = (T, d, s...) -> -ones(T, s))
 5×3 Array{Int64,2}:
  -1  -1  -1
   7  -1   4
   9  -1   6
  -1  -1  -1
   8  -1   5
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


function solve_outputs_selection(s::LogΔSizeExec, vertices::AbstractVector{<:AbstractVertex}, valuefun)
    @logmsg s.level s.msgfun(vertices[1])
    return solve_outputs_selection(base(s), vertices, valuefun)
end

solve_outputs_selection(::ΔSizeFailNoOp, vs::AbstractVector{<:AbstractVertex}, ::Any) = false, Dict(v => [1:n for n in nin(v)] for v in vs), Dict(v => 1:nout(v) for v in vs)
solve_outputs_selection(s::ThrowΔSizeFailError, vertices::AbstractVector{<:AbstractVertex}, ::Any) = throw(ΔSizeFailError(s.msgfun(vertices)))
solve_outputs_selection(s::WithValueFun, vs::AbstractVector{<:AbstractVertex}, ::Any) = solve_outputs_selection(s.strategy, vs, s.valuefun) 
"""
    solve_outputs_selection(s::AbstractΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex}, valuefun)

Returns a tuple `(success, nindict, noutdict)` where `nindict[vi]` are new input neuron indices and `noutdict[vi]` are new output neuron indices for each vertex `vi` in `vertices`.

The function generally tries to maximize `sum(valuefun(vi) .* selected[vi]) ∀ vi in vertices` where `selected[vi]` is all elements in `noutdict[vi]` larger than 0 (negative values in `noutdict` indicates a new output shall be inserted at that position). This however is up to the implementation of the `AbstractΔSizeStrategy s`.

Since selection of outputs is not guaranteed to work in all cases, a flag `success` is also returned. If `success` is `false` then applying the new indices may (and probably will) fail.
"""
function solve_outputs_selection(s::AbstractJuMPΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex}, valuefun)
    model = selectmodel(NeuronIndices(), s, vertices, values)
    case = NeuronIndices()
    # The binary variables `outselectvars` tells us which existing output indices to select
    # The integer variables `outinsertvars` tells us where in the result we shall insert -1 where -1 means "create a new output (e.g. a neuron)
    # Thus, the result will consist of all selected indices with possibly interlaced -1s

    # TODO: Cache and rescale valuefun?
    outselectvars = Dict(v => @variable(model, [1:nout(v)], Bin) for v in vertices)
    outinsertvars = Dict(v => @variable(model, [1:nout(v)], Int, lower_bound=0) for v in vertices)
    # This is the same variable name as used when adjusting size only. It allows us to delegate alot of size changing strategies to ScalarSize. 
    # Drawback is that it will yield a fair bit of back and forth so maybe it is too clever for its own good
    noutdict = Dict(v => @expression(model, sum(outselectvars[v]) + sum(outinsertvars[v])) for v in vertices)

    objexpr = sizeobjective!(case, s, vertices, (;model, outselectvars, outinsertvars, noutdict, valuefun))
    for v in vertices
        data = (;model, outselectvars, outinsertvars, noutdict, objexpr, valuefun)
        vertexconstraints!(case, v, s, data)
        objexpr = selectobjective!(case, s, v, data)
    end

    @objective(model, Max, objexpr)

    JuMP.optimize!(model)

    !accept(case, s, model) && return solve_outputs_selection(fallback(s), vertices, valuefun)

    return true, extract_ininds_and_outinds(s, outselectvars, outinsertvars)...
end

selectmodel(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, v, values) = selectmodel(case, base(s), v, values) 
selectmodel(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, v, values) = JuMP.Model(JuMP.optimizer_with_attributes(Cbc.Optimizer, "loglevel"=>0))

function vertexconstraints!(case::NeuronIndices, ::Immutable, v, s::AbstractJuMPΔSizeStrategy, data)
     @constraint(data.model, data.outselectvars[v] .== 1)
     @constraint(data.model, data.outinsertvars[v] .== 0)
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
function noinsertgaps!(model, select, insert, maxinsert=max(length(select) * 10, 200)) # TODO: get maxinsert from strategy instead
    insert_nogap = @variable(model, [1:length(insert)], Bin)

    @constraint(model, sum(insert) <= maxinsert)

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
        if nout(vi) == nin(v)[i] && offs+length(var_i)-1 <= length(var)
            @constraint(model, var_i .== var[offs:offs+length(var_i)-1])
        else
            onemismatch = true
        end
        offs += length(var_i)
    end
    return onemismatch
end

function inoutconstraint!(::NeuronIndices, s, ::SizeInvariant, v, model, vardict::Dict)
    var = vardict[v]
    onemismatch = false
    for vi in inputs(v)
        # Sizes mismatch when vertex/edge was removed (or edge added)
        if nout(vi) == nout(v)
            var_i = vardict[vi]
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
    
    scale = map(v -> max(0, maximum(data.valuefun(v))), vertices)
    noutvars = map(v -> data.noutdict[v], vertices)
    sizetargets = map(v -> nout(v) - count(<(0), data.valuefun(v)), vertices)
    # L1 norm prevents change in vertices which does not need to change.
    # Max norm tries to spread out the change so no single vertex takes most of the change.
    return norm!(SumNorm(0.1 => L1NormLinear(), 0.8 => MaxNormLinear()), data.model, @expression(data.model, objective[i=1:length(noutvars)], scale[i] * (noutvars[i] - sizetargets[i])), sizetargets)
end


function selectobjective!(case::NeuronIndices, s::AbstractJuMPΔSizeStrategy, v, data)
    value = valueobjective!(case, s, v, data)
    insertlast = insertlastobjective!(case, s, v, data)
    return @expression(data.model, data.objexpr + value + insertlast)
end
valueobjective!(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, v, data) = valueobjective!(case, base(s), v, data)
valueobjective!(::NeuronIndices, ::AbstractJuMPΔSizeStrategy, v, data) = @expression(data.model, sum(data.valuefun(v) .* data.outselectvars[v]))

insertlastobjective!(case::NeuronIndices, s::DecoratingJuMPΔSizeStrategy, v, data) = insertlastobjective!(case::NeuronIndices, base(s), v, data)
function insertlastobjective!(::NeuronIndices, s, v, data)
    insvars = data.outinsertvars[v]
    
    isempty(insvars) && return @expression(data.model, 0)

    preferend = collect(length(insvars) : -1 : 1)
    scale = minimum(x -> x >= 0 ? x : typemax(x), data.valuefun(v)) / sum(preferend) 
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


