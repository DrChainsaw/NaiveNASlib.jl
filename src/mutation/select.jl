
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
    AbstractJuMPSelectionStrategy

Base type for how to select the exact inputs/outputs indices from a vertex given a size change using JuMP to handle the constraints.
"""
abstract type AbstractJuMPSelectionStrategy <: AbstractSelectionStrategy end

fallback(::AbstractJuMPSelectionStrategy) = SelectionFail()

"""
    NoutExact <: AbstractSelectionStrategy
    NoutExact()
    NoutExact(fallbackstrategy)

Selects output indices from a vertex with the constraint that `nout(v)` for all `v` which needs to change as a result of the selection are unchanged.

Possible to set a fallbackstrategy should it be impossible to select indices according to the strategy.
"""
struct NoutExact <: AbstractJuMPSelectionStrategy
    fallback::AbstractSelectionStrategy
end
NoutExact() = NoutExact(LogSelectionFallback("Relaxing size constraint...", NoutRelaxSize()))
fallback(s::NoutExact) = s.fallback

"""
    NoutRelaxSize <: AbstractSelectionStrategy
    NoutRelaxSize()
    NoutRelaxSize(lower, upper)
    NoutRelaxSize(fallbackstrategy)
    NoutRelaxSize(lower, upper, fallbackstrategy)

Selects output indices from a vertex with the constraint that `nout(v)` for all vertices `v` which needs to change as a result of the selection is in the range `max(1, lower * noutv) <= nout(v) <= upper * noutv` where `noutv` is the result of `nout(v)` before selecting indices.

Possible to set a fallbackstrategy should it be impossible to select indices according to the strategy.
"""
struct NoutRelaxSize <: AbstractJuMPSelectionStrategy
    lower::Real
    upper::Real
    fallback::AbstractSelectionStrategy
end
NoutRelaxSize(fallback=LogSelectionFallback("Reverting...", NoutRevert())) = NoutRelaxSize(0.7, 1, NoutRelaxSize(0.5, 1, NoutRelaxSize(0.3, 1.5, NoutRelaxSize(0.2, 2, fallback))))
NoutRelaxSize(lower::Real, upper::Real) = NoutRelaxSize(lower, upper, NoutRevert())
fallback(s::NoutRelaxSize) = s.fallback

"""
    NoutMainVar <: AbstractJuMPSelectionStrategy
    NoutMainVar()
    NoutMainVar(main::AbstractJuMPSelectionStrategy, child::AbstractJuMPSelectionStrategy)

Adds size constraints also for main variables while allowing other (typically relaxed) constraints for children (i.e vertices which will be changed as a consequence of changing the main vertex).

Typically used after making a structural mutation so that part of the graph will be barred from size changes.

Possible to set fallbackstrategies for both main and children. Note that with the exception of `LogSelection` only instances of `AbstractJuMPSelectionStrategy` are allowed for the child fallback strategy.
"""
struct NoutMainVar <: AbstractJuMPSelectionStrategy
    main::AbstractJuMPSelectionStrategy
    child::AbstractJuMPSelectionStrategy
end
NoutMainVar() = NoutMainVar(NoutExact(), NoutRelaxSize())
NoutMainVar(m::LogSelection, c) = LogSelection(m.level, m.msgfun, NoutMainVar(m.andthen, c))
NoutMainVar(m::AbstractSelectionStrategy, c) = m
fallback(s::NoutMainVar) = NoutMainVar(fallback(s.main), fallback(s.child))


struct ValidOutsInfo{I <:Integer, T <: MutationTrait}
    current::Matrix{I}
    after::Matrix{I}
    trait::T
end
ValidOutsInfo(currsize::I, aftersize::I, trait::T) where {T<:MutationTrait,I<:Integer} = ValidOutsInfo(zeros(I, currsize, 0), zeros(I, aftersize, 0), trait)
addinds(v::ValidOutsInfo{I, T}, c::Integer, a::Integer) where {I,T} = ValidOutsInfo([v.current range(c, length=size(v.current,1))], [v.after range(a, length=size(v.after,1))], v.trait)


"""
    validouts(v::AbstractVertex)

Return a `Dict` mapping vertices `vi` seen from `v`s output direction to `ValidOutsInfo mi`.

For output selection to be consistent, either all or none of the indices in a row in `mi.current` must be selected.

For output insertion to be consistent, either all or none of the indices in a row in `mi.after` must be chosen.

 Matrices in `mi` have more than one column if activations from `vi` is input (possibly through an arbitrary number of size transparent layers such as BatchNorm) to `v` more than once.

 Furthermore, in the presense of `SizeInvariant` vertices, some indices may be present in more than one `mi`, making selection of indices non-trivial at best in the general case.

 # Examples
 ```julia-repl
julia> iv = inputvertex("in", 2, FluxDense());

julia> v1 = mutable("v1", Dense(2, 3), iv);

julia> v2 = mutable("v2", Dense(2, 5), iv);

julia> v = concat(v1,v2,v1,v2);

julia> cdict = validouts(v);

julia> nout(v)
16

julia> for (vi, mi) in cdict
       @show name(vi)
       display(mi.current)
       end
name(vi) = "v2"
5×2 Array{Int64,2}:
 4  12
 5  13
 6  14
 7  15
 8  16
name(vi) = "v1"
3×2 Array{Int64,2}:
 1   9
 2  10
 3  11
```

"""
validouts(v::AbstractVertex, skipin::Set, skipout::Set, out::Bool=true) = validouts(v, out, v, Dict(), (), (1, 1), maskinit(v), Set(), skipin, skipout)
function validouts(v::AbstractVertex, out::Bool=true, vfrom::AbstractVertex=v, dd=Dict(), path=(), offs=(1, 1), mask=maskinit(v), visited = Set(), skipin::Set=Set(), skipout::Set=Set())
    has_visited!(visited, v) && return dd
    out && v in skipin && return dd
    !out && v in skipout && return dd
    validouts(v, Val(out), vfrom, dd, path, offs, mask, visited, skipin, skipout)
    delete!(visited, v)
    return dd
end
maskinit(v) = (trues(nout_org(v)), trues(nout(v)))
validouts(v::AbstractVertex, args...) = validouts(trait(v), v, args...)
validouts(t::DecoratingTrait, args...) = validouts(base(t), args...)

function validouts(::SizeStack, v, ::Val{true}, vfrom, dd, path, offs, mask, args...)
    cnts = (1, 1)
    for vin in inputs(v)
        next = cnts .+ (nout_org(vin), nout(vin)) .- 1
        newmask = map(mt -> mt[1][mt[2]:mt[3]], zip(mask, cnts, next))
        cnts = cnts .+ (nout_org(vin), nout(vin))
        validouts(vin, true, v, dd, path, offs, newmask, args...)
        offs = offs .+ sum.(newmask)
    end
    return dd
end

function validouts(::SizeStack, v, ::Val{false}, vfrom, dd, path, offs, mask, args...)
    newmask = (BitVector(), BitVector())
    for vin in inputs(v)
        if vin == vfrom
            append!.(newmask, mask)
        else
            append!.(newmask, map(mv -> .!mv, maskinit(vin)))
        end
    end

    for (p, vout) in enumerate(outputs(v))
        newpath = (path..., p)
        validouts(vout, false, v, dd, newpath, offs, newmask, args...)
    end
    return dd
end

function validouts(::SizeInvariant, v, out, vfrom, dd, path, args...)
    for (p, vout) in enumerate(outputs(v))
        newpath = (path..., p)
        validouts(vout, false, v, dd, newpath, args...)
    end
    foreach(vin -> validouts(vin, true, v, dd, path, args...), inputs(v))
    return dd
end

validouts(t::SizeAbsorb, args...) = addvalidouts(t, args...)
validouts(t::Immutable, args...) = addvalidouts(t, args...)

function addvalidouts(t::MutationTrait, v, ::Val{true}, vfrom, dd, path, offs, mask, visited, args...)
    initial = length(visited) == 1

    # length(visited) > 1 is only false if the first vertex we call validouts for is of type SizeAbsorb
    # If it is, we need to propagate the call instead of adding indices as the outputs of v might take us to a SizeInvariant vertex which in turn might take us to a SizeStack vertex
    orgsize = sum(mask[1])
    newsize = sum(mask[2])
    if !initial
        info = get(() -> ValidOutsInfo(orgsize, newsize, t), dd, v)
        dd[v] = addinds(info, offs...)
    end
    for (p, vout) in enumerate(outputs(v))
        newpath = (path..., p)
        validouts(vout, false, v, dd, newpath, offs, mask, visited, args...)
    end

    # This is true if all outputs of v also are (or lead to) size absorb types and we shall indeed populate dd with the indices of this vertex
    if initial && isempty(dd)
        dd[v] = addinds(ValidOutsInfo(orgsize, newsize, t), offs...)
    end

    return dd
end
function addvalidouts(t::MutationTrait, v, ::Val{false}, vfrom, dd, args...)
    foreach(vin -> validouts(vin, true, v, dd, args...), inputs(v))
    return dd
end

function has_visited!(visited, x)
    x in visited && return true
    push!(visited, x)
    return false
end

"""
    select_outputs(v::AbstractVertex, values)
    select_outputs(s::AbstractSelectionStrategy, v, values)

Returns a tuple `(success, result)` where `result` is a vector so that `Δnout(v, result)` selects outputs for `v` in a way which is consistent with (or as close as possible to) current output size for all vertices visible in the out direction of `v`.

The function generally tries to maximize the `sum(values[selected])` where `selected` is all elements in `results` larger than 0 (negative values in `result` indicates a new output shall be inserted at that position). This however is up to the implementation of the `AbstractSelectionStrategy s`.

Since selection of outputs is not guaranteed to work in all cases, a flag `success` is also returned. If `success` is `false` then calling `Δnout(v, result)` might fail.

See [`validouts`](@ref) for a description of the constraints which may cause the selection to fail.

# Examples
```julia-repl
julia> iv = inputvertex("in", 2, FluxDense());

julia> v1 = mutable("v1", Dense(2, 3), iv);

julia> v2 = mutable("v2", Dense(2, 5), iv);

julia> v = concat(v1,v2,v1,v2);

julia> Δnout(v, -2);

julia> nout(v1)
3

julia> nout(v2)
4

julia> NaiveGAflux.select_outputs(v, 1:nout_org(op(v))) # Dummy values, prefer the higher indices
(true, [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16])

julia> Δnout(v1, 3);

julia> NaiveGAflux.select_outputs(v, 1:nout_org(op(v)))
(true, [1, 2, 3, -1, -1, -1, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, 13, 14, 15, 16])
```

"""
select_outputs(v::AbstractVertex, values, skipin=[], skipout=[]) = select_outputs(NoutExact(), v, values, skipin, skipout)
select_outputs(s::AbstractSelectionStrategy, v, values, skipin=[], skipout=[])= select_outputs(s, v, values, validouts(v, Set(skipin), Set(skipout)))

function select_outputs(s::LogSelection, v, values, cdict)
    @logmsg s.level s.msgfun(v)
    return select_outputs(s.andthen, v, values, cdict)
end

select_outputs(s::SelectionFail, v, values, cdict) = error("Selection failed for vertex $(name(v))")

function select_outputs(s::NoutRevert, v, values, cdict)
    if !ismissing(minΔnoutfactor(v))
        Δ = nout_org(v) - nout(v)
        Δnout(v, Δ)
    end
    return false, 1:nout(v)
end

function select_outputs(s::AbstractJuMPSelectionStrategy, v, values, cdict)
    model = selectmodel(s, v, values)

    # Select indices using integer linear programming.
    # High level description of formulation:
    # `values` is the metric which we are out to maximize (although in the general case when things are engangled we're actually mostly interested in just getting any feasible solution)

    # The binary variable `selectvar` tells us which indices of `values` to select
    # The binary variable `inservar` tells us where in the result we shall insert -1 where -1 means "create a new output (e.g. a neuron)
    # Thus, the result will consist of all selected indices with possibly interlaced -1s

    # Now, the constraints are the following three types:
    # 1. Row constraint: The values of `cdict` has matrices consisting of subsets of the indices in selectvar and insertvar respectively. The constraint is that for selection to be consistent, either all or none of the indices in each row in those matrices must be selected. Note that this applies to both selection and insertion. See documentation of validouts for a description of what the matrices represent.
    # 2. Size constraint: Much more straighforward! The for each key `vi` of type `AbstractVertex` in `cdict`, select `min(nout(vi), nout_org(vi))` (i.e either all or a subset) of the rows in `mi.current` where `mi.current` is the selection matrix mapped to `vi`. Similarly, the number of rows in `mi.after` minus the number of selected rows in `mi.after` shall be equal to the number of inserted indices. Note that size constraint can be relaxed, meaning that more or fewer indices are selected from each row.
    # 3. Δfactor constraint: This constraint makes sure minΔnoutfactors are respected, i.e any selection+insertion of outputs must result in a change which is an integer multiple of the associated minΔnoutfactor.

    # variable for selecting a subset of the existing outputs.
    selectvar = @variable(model, selectvar[1:length(values)], Bin)
    # Variable for deciding at what positions to insert new outputs.
    insertvar = @variable(model, insertvar[1:nout(v)], Bin)

    # insertlast will be added to objective to try to insert new neurons last
    # Should not really matter whether it does that or not, but makes things a bit easier to debug
    insertlast = sizeconstraintmainvar(s, SizeAbsorb(), v, model, selectvar, insertvar)

    for (vi, mi) in cdict
        # TODO: Don't add to cdict if this happens?
        isempty(mi.current) && continue
        select_i = rowconstraint(s, model, selectvar, mi.current)
        sizeconstraint(s, mi.trait, vi, model, select_i)
        Δsizeexp = @expression(model, length(select_i) - sum(select_i))

        # Check if size shall be increased
        if length(insertvar) > length(selectvar)
            # This is a bit unfortunate as it basically makes it impossible to relax
            # Root issue is that mi.after is calculated under an the assumption that current nout(vi) will be the size after selection as well. Maybe some other formulation does not have this restricion, but I can't come up with one which is guaranteed to either work or know it has failed.
            insert_i = rowconstraint(s, model, insertvar, mi.after)
            Δsizeexp = @expression(model, Δsizeexp + sum(insert_i))

            @constraint(model, insert_i[1] == 0) # Or else it won't be possible to know where to split
            # This will make sure that we are consistent to what mi.after prescribes
            # Note that this basically prevents relaxation of size constraint, but this is needed because mi.after is calculated assuming nout(vi) is the result after selection.
            # It does offer the flexibility to trade an existing output for a new one should that help resolving something.
            @constraint(model, sum(insert_i) == length(insert_i) - sum(select_i))

            last_i = min(length(select_i), length(insert_i))
            insertlast = @expression(model, insertlast + sum(insert_i[1:last_i]))
        end

        Δfactorconstraint(s, model, minΔnoutfactor(vi), Δsizeexp)

    end

    @objective(model, Max, values' * selectvar - insertlast)

    JuMP.optimize!(model)

    !accept(s, model) && return select_outputs(fallback(s), v, values, cdict)

    # insertvar is 1.0 at indices where a new output shall be added and 0.0 where an existing one shall be selected
    result = -round.(Int, JuMP.value.(insertvar))
    selected = findall(xi -> xi > 0, JuMP.value.(selectvar))

    # TODO: Needs investigation
    sum(result) == 0 && return true, selected

    j = 1
    for i in eachindex(result)
        if result[i] == 0
            result[i] = selected[j]
            j += 1
        end
    end

    return true, result
end

accept(::AbstractJuMPSelectionStrategy, model::JuMP.Model) = JuMP.termination_status(model) != MOI.INFEASIBLE && JuMP.primal_status(model) == MOI.FEASIBLE_POINT # Beware: primal_status seems unreliable for Cbc. See MathOptInterface issue #822

selectmodel(::AbstractJuMPSelectionStrategy, v, values) = JuMP.Model(JuMP.with_optimizer(Cbc.Optimizer, loglevel=0))

function rowconstraint(::AbstractJuMPSelectionStrategy, model, x, indmat)
    # Valid rows don't include all rows when vertex to select from has not_org < nout_org of a vertex in cdict.
    # This typically happens when resizing vertex to select from due to removal of its output vertex
    validrows = [i for i in 1:size(indmat, 1) if all(indmat[i,:] .<= length(x))]
    var = @variable(model, [1:length(validrows)], Bin)
    @constraint(model, size(indmat,2) .* var .- sum(x[indmat[validrows,:]], dims=2) .== 0)
    return var
end


sizeconstraintmainvar(::AbstractJuMPSelectionStrategy, t, v, model, selvar, insvar) = @expression(model, 0)
function sizeconstraintmainvar(s::NoutMainVar, t, v, model, selvar, insvar)
    toselect = min(length(selvar), length(insvar))
    sizeconstraint(s.main, toselect, model, selvar)
    if length(insvar) > length(selvar)
        @constraint(model, sum(insvar) == length(insvar) - sum(selvar))
    end
    return @expression(model, sum(insvar[1:toselect]))
end

function sizeconstraint(s::NoutMainVar, t, v, model, var)
    #sizeconstraint(s.child, t, v, model, var)
end

nselect_out(v) = min(nout(v), nout_org(v))
limits(s::NoutRelaxSize, n) =  (max(1, s.lower * n), s.upper * n)

sizeconstraint(s::AbstractJuMPSelectionStrategy, t, v, model, var) = sizeconstraint(s, min(length(var), nselect_out(v)), model, var)
sizeconstraint(s::NoutRelaxSize, t::Immutable, v, model, var) = sizeconstraint(NoutExact(), t, v, model, var)

sizeconstraint(::NoutExact, size::Integer, model, var) = @constraint(model, sum(var) == size)
function sizeconstraint(s::NoutRelaxSize, sizetarget::Integer, model, var)
    # Wouldn't mind being able to relax the size constraint like this:
    #@objective(model, Min, 10*(sum(selectvar) - sizetarget^2))
    # but that requires MISOCP and only commercial solvers seem to do that
    nmin, nmax = limits(s, sizetarget)
    @constraint(model, nmin <= sum(var) <= nmax)
end

Δfactorconstraint(s::NoutMainVar, model, f, Δsizeexp) = Δfactorconstraint(s.child, model, f, Δsizeexp)
function Δfactorconstraint(::NoutExact, model, f, Δsizeexp) end
function Δfactorconstraint(::NoutRelaxSize, model, ::Missing, Δsizeexp) end
function Δfactorconstraint(::NoutRelaxSize, model, f, Δsizeexp)
    # Δfactor constraint:
    #  - Constraint that answer shall result in a size change which is an integer multiple of f.
    fv = @variable(model, integer=true)
    @constraint(model, f * fv == Δsizeexp)
end


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
    DefaultJuMPSelectionStrategy <: AbstractJuMPSelectionStrategy

Default strategy intended to be used when adding some extra constraints or objectives to a model on top of the default.
"""
struct DefaultJuMPSelectionStrategy <: AbstractJuMPSelectionStrategy end

"""
    Δoutputs(v::AbstractVertex, valuefun::Function)
    Δoutputs(s::AbstractSelectionStrategy, v::AbstractVertex, valuefun::Function)
    Δoutputs(d::Direction, v::AbstractVertex, valuefun::Function)
    Δoutputs(s::AbstractSelectionStrategy, d::Direction, v::AbstractVertex, valuefun::Function)

Change outputs of `v` according to the provided `AbstractSelectionStrategy s` (default `OutSelect{Exact}`).

Argument `valuefun` provides a vector `value = valuefun(vx)` for any vertex `vx` in the same graph as `v` where `value[i] > value[j]` indicates that output index `i` shall be preferred over `j` for vertex `vx`.

If provided, `Direction d` will narrow down the set of vertices to evaluate so that only vertices which may change as a result of changing size of `v` are considered.
"""
Δoutputs(v::AbstractVertex, valuefun::Function) = Δoutputs(OutSelectExact(), v, valuefun)
Δoutputs(s::AbstractSelectionStrategy, v::AbstractVertex, valuefun::Function) = Δoutputs(s, all_in_graph(v), valuefun)
function Δoutputs(s::SelectDirection, v::AbstractVertex, valuefun::Function)
    nin_change = nin_org(v) != nin(v)
    nout_change = nout(v) != nout(v)
    if nout_change && nin_change
        Δoutputs(s.strategy, Both(), v, valuefun)
    elseif nout_change
        Δoutputs(s.strategy, Output(), v, valuefun)
    elseif nin_change
        Δoutputs(s.strategy, Input(), v, valuefun)
    end
 end

Δoutputs(d::Direction, v::AbstractVertex, valuefun::Function) = Δoutputs(OutSelectExact(), d, v, valuefun)
Δoutputs(s::AbstractSelectionStrategy, d::Direction, v::AbstractVertex, valuefun::Function) = Δoutputs(s, all_in_Δsize_graph(v, d), valuefun)

function Δoutputs(s::AbstractSelectionStrategy, vs::AbstractVector{<:AbstractVertex}, valuefun::Function)
    success, ins, outs = solve_outputs_selection(s, vs, valuefun)
    if success
        Δoutputs(ins, outs, vs)
    end
end

"""
    Δoutputs(ins::Dict outs::Dict, vertices::AbstractVector{<:AbstractVertex})

Set input and output indices of each `vi` in `vs` to `outs[vi]` and `ins[vi]` respectively.
"""
function Δoutputs(ins::Dict, outs::Dict, vs::AbstractVector{<:AbstractVertex})

    for vi in vs
        Δnin_no_prop(vi, ins[vi]...)
        Δnout_no_prop(vi, outs[vi])
    end

    for vi in vs
        after_Δnin(vi, ins[vi]...)
        after_Δnout(vi, outs[vi])
    end
end

function Δnin_no_prop(v) end
function Δnin_no_prop(v, inds::Missing) end
function Δnin_no_prop(v, inds::AbstractVector{<:Integer}...)
    any(inds .!= [1:insize for insize in nin(v)]) || return
    Δnin_no_prop(trait(v), v, inds)
end

function Δnout_no_prop(v, inds::Missing) end
function Δnout_no_prop(v, inds::AbstractVector{<:Integer})
    inds == 1:nout(v) && return
    Δnout_no_prop(trait(v), v, inds)
end


function solve_outputs_selection(s::LogSelection, vertices::Vector{AbstractVertex}, valuefun::Function)
    @logmsg s.level s.msgfun(vertices[1])
    return solve_outputs_selection(s.andthen, vertices, valuefun)
end

solve_outputs_selection(s::SelectionFail, vertices::Vector{AbstractVertex}, valuefun::Function) = error("Selection failed for vertex $(name.(vertices))")

function solve_outputs_selection(s::NoutRevert, vertices::Vector{AbstractVertex}, valuefun::Function)
    for v in vertices
        diff = nout_org(v) - nout(v)
        if diff != 0
            Δnout(v, diff)
        end
    end

    return false, Dict(vertices .=> UnitRange.(1, nout.(vertices))), Dict(vertices .=> map(nins -> UnitRange.(1,nins), nin.(vertices)))
end

function solve_outputs_selection(s::AbstractJuMPSelectionStrategy, vertices::Vector{AbstractVertex}, valuefun::Function)
    model = selectmodel(s, vertices, values)

    # The binary variables `outselectvars` tells us which existing output indices to select
    # The binary variables `outinsertvars` tells us where in the result we shall insert -1 where -1 means "create a new output (e.g. a neuron)
    # Thus, the result will consist of all selected indices with possibly interlaced -1s

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
    inoutconstraint!(s, t, v, data)
end

function sizeconstraint!(::OutSelect{Exact}, t, v, data)
    @constraint(data.model, sum(data.outselectvars[v]) == nselect_out(v))
    sizeconstraint!(DefaultJuMPSelectionStrategy(), t, v, data)
end

function sizeconstraint!(::AbstractJuMPSelectionStrategy, t, v, data)
    # Handle insertions
    # The constraint that either there are no new outputs or the total number of outputs must be equal to the length of outinsertvars is a somewhat unfortunate result of the approach chosen to solve the problem.
    # If not enforced, we will end up in situations where some indices shall neither be selected nor have insertions. For example, the result might say "keep indices 1,2,3 and insert a new output at index 10".
    # If one can come up with a constraint to formulate "no gaps" (such as the gab above) instead of the current approach the chances of finding a feasible soluion would probably increase.
    # Maybe this https://cs.stackexchange.com/questions/12102/express-boolean-logic-operations-in-zero-one-integer-linear-programming-ilp in combination with this https://math.stackexchange.com/questions/2022967/how-to-model-a-constraint-of-consecutiveness-in-0-1-programming?rq=1
    outsel = data.outselectvars[v]
    outins = data.outinsertvars[v]
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
