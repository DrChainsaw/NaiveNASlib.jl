

nin(v::AbstractVertex) = nin(trait(v), v)
nin(t::DecoratingTrait, v::AbstractVertex) = nin(base(t), v)
nin(t::MutationTrait, v::AbstractVertex, vo::AbstractVertex=v) = nin(t, base(v), vo)
nin(t::MutationTrait, v::CompVertex, vo::AbstractVertex) = nin(v.computation, t, vo)
nin(::MutationTrait, ::InputSizeVertex) = []

# Give users a chance to fallback to smoe generic property of the vertex
nin(f, ::MutationTrait, ::AbstractVertex) = nin(f)
nin(f, ::SizeTransparent, v::AbstractVertex) = nout.(inputs(v))

nout(v::AbstractVertex) = nout(trait(v), v)
nout(t::DecoratingTrait, v::AbstractVertex) = nout(base(t), v)
nout(t::MutationTrait, v::AbstractVertex, vo::AbstractVertex=v) = nout(t, base(v), vo)
nout(t::MutationTrait, v::CompVertex, vo::AbstractVertex) = nout(v.computation, t, vo)
nout(::MutationTrait, v::InputSizeVertex) = v.size

# Give users a chance to fallback to smoe generic property of the vertex or trait
nout(f, ::MutationTrait, ::AbstractVertex) = nout(f)

# SizeTransparent might not care about size
nout(f, ::SizeInvariant, v::AbstractVertex) = isempty(nin(v)) ? 0 : nin(v)[1]
nout(f, ::SizeStack, v::AbstractVertex) = isempty(nin(v)) ? 0 : sum(nin(v))

# TODO: Remove
nout_org(v::AbstractVertex) = nout(v)
nin_org(v::AbstractVertex) = nin(v)

# Add possibly for shape traits for computations
nout(f) = nout(shapetrait(f), f)
nin(f) = nin(shapetrait(f), f)

shapetrait(f::T) where T = T

struct NoShapeTraitError <: Exception
    msg::String
end
Base.showerror(io::IO, e::NoShapeTraitError) = print(io, e.msg)

noutnin_errmsg(f, name) = "$name not defined for $(f)! Either implement NaiveNASlib.$name(::$(typeof(f))) or implement NaiveNASlib.$name(::ST, ::$(typeof(f))) where ST is the return type of NaiveNASlib.shapetrait(::$(typeof(f)))" 

nout(::Any, f) = throw(NoShapeTraitError(noutnin_errmsg(f, nout)))
nin(::Any, f) = throw(NoShapeTraitError(noutnin_errmsg(f, nin)))

"""
    minΔnoutfactor(v::AbstractVertex)

Returns the smallest `k` so that allowed changes to `nout` of `v` as well as `nin` of its outputs are `k * n` where `n` is an integer.
Returns `missing` if it is not possible to change `nout`.
"""
minΔnoutfactor(v::AbstractVertex) = missing
minΔnoutfactor(v::MutationVertex) = minΔnoutfactor(trait(v), v)
minΔnoutfactor(t::DecoratingTrait, v::AbstractVertex) = minΔnoutfactor(base(t), v)
minΔnoutfactor(::MutationTrait, v::AbstractVertex) = lcmsafe(vcat(minΔninfactor_only_for.(outputs(v))..., minΔnoutfactor_only_for(v)))
minΔnoutfactor(t::SizeTransparent, v::AbstractVertex) = minΔninfactor(t, v)

"""
    minΔnoutfactor(v::AbstractVertex, [s=VisitState{Int}()])

Returns the smallest `k` so that allowed changes to `nin` of `v` as well as `nout` of its inputs are `k * n` where `n` is an integer.
Returns `missing` if it is not possible to change `nin`.
"""
minΔninfactor(v::AbstractVertex) = lcmsafe(vcat(minΔnoutfactor_only_for.(inputs(v))..., minΔninfactor_only_for(v)))
minΔninfactor(v::MutationVertex) = minΔninfactor(trait(v), v)
minΔninfactor(t::DecoratingTrait, v::AbstractVertex) = minΔninfactor(base(t), v)
minΔninfactor(::MutationTrait, v::AbstractVertex) = lcmsafe(vcat(minΔnoutfactor_only_for.(inputs(v))..., minΔninfactor_only_for(v)))
minΔninfactor(::SizeTransparent, v::AbstractVertex) = lcmsafe([minΔnoutfactor_only_for(v), minΔninfactor_only_for(v)])


"""
    minΔnoutfactor_only_for(v::AbstractVertex, [s=VisitState{Int}()])

Returns the smallest `k` so that allowed changes to `nout` of `v` are `k * n` where `n` is an integer.
Returns `missing` if it is not possible to change `nout`.
"""
minΔnoutfactor_only_for(v::AbstractVertex, s=[]) = outvisit(v, s) ? 1 : minΔnoutfactor_only_for(base(v),s)
minΔnoutfactor_only_for(v::MutationVertex, s=[]) = outvisit(v, s) ? 1 : minΔnoutfactor_only_for(trait(v), v, s)
minΔnoutfactor_only_for(v::InputVertex, s=[]) = missing
minΔnoutfactor_only_for(v::CompVertex, s=[]) = minΔnoutfactor(v.computation)
minΔnoutfactor(f::Function) = 1 # TODO: Move to test as this does not make alot of sense
minΔnoutfactor_only_for(t::DecoratingTrait, v::AbstractVertex, s) = minΔnoutfactor_only_for(base(t), v, s)
minΔnoutfactor_only_for(::Immutable, v::AbstractVertex, s) = missing
minΔnoutfactor_only_for(::SizeAbsorb, v::AbstractVertex, s) = minΔnoutfactor_only_for(base(v),s)

minΔnoutfactor_only_for(::SizeInvariant, v::AbstractVertex, s) = lcmsafe(vcat(map(vi -> minΔnoutfactor_only_for(vi,s), inputs(v)), map(vo -> minΔninfactor_only_for(vo,s), outputs(v))))
function minΔnoutfactor_only_for(::SizeStack, v::AbstractVertex, s)
    absorbing = findterminating(v, inputs, outputs)

    # This is not strictly the minimum as using only one of the factors would work as well
    # However, this would create a bias as the same factor would be used all the time
    # Life is really hard sometimes :(

    # Count thingy is for duplicate outputs. Must be counted twice as it is impossible
    # to only change one of them, right?
    factors = [count(x->x==va,absorbing) * minΔnoutfactor_only_for(va,s) for va in unique(absorbing)]
    return lcmsafe(factors)
end

"""
    minΔninfactor_only_for(v::AbstractVertex)

Returns the smallest `k` so that allowed changes to `nin` of `v` are `k * n` where `n` is an integer.
Returns `missing` if it is not possible to change `nin`.
"""
minΔninfactor_only_for(v::AbstractVertex, s=[]) = invisit(v, s) ? 1 : minΔninfactor_only_for(base(v),s)
minΔninfactor_only_for(v::MutationVertex, s=[]) = invisit(v, s) ? 1 : minΔninfactor_only_for(trait(v), v, s)
minΔninfactor_only_for(v::InputVertex,s=[]) = missing
minΔninfactor_only_for(v::CompVertex,s=[]) = minΔninfactor(v.computation)
minΔninfactor(f::Function) = 1 # TODO: Move to test as this does not make alot of sense
minΔninfactor_only_for(t::DecoratingTrait, v::AbstractVertex, s) = minΔninfactor_only_for(base(t), v, s)
minΔninfactor_only_for(::Immutable, v::AbstractVertex, s) = missing
minΔninfactor_only_for(::SizeAbsorb, v::AbstractVertex, s) = minΔninfactor_only_for(base(v), s)
minΔninfactor_only_for(t::SizeInvariant, v::AbstractVertex, s) = minΔnoutfactor_only_for(t, v, s)
function minΔninfactor_only_for(::SizeStack, v::AbstractVertex, s)
    absorbing = findterminating(v, outputs, inputs)
    # This is not strictly the minimum as using only one of the factors would work as well
    # However, this would create a bias as the same factor would be used all the time
    # Life is really hard sometimes :(

    # Count thingy is for duplicate inputs. Must be counted twice as it is impossible
    # to only change one of them, right?
    factors = [count(x->x==va,absorbing) * minΔnoutfactor_only_for(va, s) for va in unique(absorbing)]
    return lcmsafe(factors)
end

#lcm which also checks for missing and arrays of undefined type
function lcmsafe(x)
    isempty(x) && return 1
    return any(ismissing.(x)) ? missing : lcm(Integer.(x))
end

invisit(v, visited) = hasvisited(v, Input(), visited)
outvisit(v, visited) = hasvisited(v, Output(), visited)

function hasvisited(v, d::Direction, visited)
    (v, d) in visited && return true
    push!(visited, (v, d))
    return false
end


"""
    findterminating(v::AbstractVertex, direction::Function, other::Function= v -> [], visited = [])

Return an array of all vertices which terminate size changes (i.e does not propagate them) seen through the given direction (typically inputs or outputs). A vertex will be present once for each unique path through which its seen.

The `other` direction may be specified and will be traversed if a SizeInvariant vertex is encountered.

Will return the given vertex if it is terminating.

# Examples
```julia-repl

julia> v1 = inputvertex("v1", 3);

julia> v2 = inputvertex("v2", 3);

julia> v3 = conc(v1,v2,v1,dims=1);

julia> name.(findterminating(v1, outputs, inputs))
1-element Array{String,1}:
 "v1"

julia> name.(findterminating(v3, outputs, inputs))
0-element Array{Any,1}

julia> name.(findterminating(v3, inputs, outputs))
3-element Array{String,1}:
 "v1"
 "v2"
 "v1"

 julia> v5 = v3 + inputvertex("v4", 9);

 julia>  # Note, + creates a SizeInvariant vertex and this causes its inputs to be seen through the output direction

 julia> name.(findterminating(v3, outputs, inputs))
 1-element Array{String,1}:
  "v4"
```
"""
function findterminating(v::AbstractVertex, direction::Function, other::Function=v->[], visited = Set{AbstractVertex}())
    v in visited && return []
    push!(visited, v)
    res = findterminating(trait(v), v, direction, other, visited)
    delete!(visited, v)
    return res
 end
findterminating(t::DecoratingTrait, v, d::Function, o::Function, visited) = findterminating(base(t), v, d, o, visited)
findterminating(::SizeAbsorb, v, d::Function, o::Function, visited) = [v]
findterminating(::Immutable, v, d::Function, o::Function, visited) = [v]

findterminating(::SizeStack, v, d::Function, o::Function, visited) = collectterminating(v, d, o, visited)
findterminating(::SizeInvariant, v, d::Function, o::Function, visited) = vcat(collectterminating(v, d, o, visited), collectterminating(v, o, d, visited))
collectterminating(v, d::Function, o::Function, visited) = mapfoldl(vf -> findterminating(vf, d, o, visited), vcat, d(v), init=[])

"""
    ScalarSize
    
Treat vertices as having a scalar size when formulating the size change problem.
"""
struct ScalarSize end

"""
    Δsize!(case, s::AbstractΔSizeStrategy, vertices::AbstractArray{<:AbstractVertex})

Calculate new sizes for (potentially) all provided `vertices` using the strategy `s` and apply all changes.
"""
function Δsize!(case::ScalarSize, s::AbstractΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex})
    execute, nins, nouts = newsizes(s, vertices)
    if execute
        Δsize!(case, nins, nouts, vertices)
    end
    return execute
end

"""
    Δsize!(nins::AbstractDict, nouts::AbstractVector{<:Integer}, vertices::AbstractVector{<:AbstractVertex})

Set output size of `vertices[i]` to `nouts[i]` for all `i` in `1:length(vertices)`.
Set input size of all keys `vi` in `nins` to `nins[vi]`.
"""
function Δsize!(case::ScalarSize, nins::AbstractDict, nouts::AbstractVector, vertices::AbstractVector{<:AbstractVertex})

    Δnouts = nouts .- nout.(vertices)
    Δnins = [coalesce.(get(() -> nin(vi), nins, vi), nin(vi)) .- nin(vi) for vi in vertices]

    for (i, vi) in enumerate(vertices)
        insizes = get(() -> nin(vi), nins, vi)
        insizes = coalesce.(insizes, nin(vi))
        Δsize!(case, OnlyFor(), vi, insizes, nouts[i])
    end

    for (i, vi) in enumerate(vertices)
        after_Δnin(vi, Δnins[i], any(!=(0), Δnins[i]))
        after_Δnout(vi, Δnouts[i], Δnouts[i] != 0)
    end
end

Δsize!(case::ScalarSize, s::OnlyFor, v::AbstractVertex, insizes::AbstractVector{<:Integer}, outsize::Integer) = Δsize!(case, s, base(v), insizes, outsize)
Δsize!(::ScalarSize, ::OnlyFor, v::CompVertex, insizes::AbstractVector{<:Integer}, outsize::Integer) = Δsize!(v.computation, insizes, outsize)
function Δsize!(f, ins::AbstractVector{<:Integer}, outs::Integer) end

function Δsize!(::ScalarSize, ::OnlyFor, v::InputSizeVertex, insizes::AbstractVector{<:Integer}, outs::Integer) 
    if !all(isempty, skipmissing(ins))
        throw(ArgumentError("Try to change input size of InputVertex $(name(v)) to $insizes"))
    end 
    if outsize != nout(v)
        throw(ArgumentError("Try to change output size of InputVertex $(name(v)) to $outsize"))
    end
end

after_Δnin(v, Δs, changed) = after_Δnin(trait(v), v, Δs, changed)
after_Δnin(t::DecoratingTrait, v, Δs, changed) = after_Δnin(base(t), v, Δs, changed)
function after_Δnin(t::SizeChangeValidation, v, Δs, changed) 
    after_Δnin(base(t), v, Δs, changed)
    validate_Δnin(v, Δs)
end
function after_Δnin(t::SizeChangeLogger, v, Δs, changed) 
    if changed
        @logmsg t.level "Change nin of $(infostr(t, v)) by $(join(compressed_string.(Δs), ", ", " and "))"
    end
    after_Δnin(base(t), v, Δs, changed)
end
function after_Δnin(t, v, Δs, changed) end

after_Δnout(v, Δ, changed) = after_Δnout(trait(v), v, Δ, changed)
after_Δnout(t::DecoratingTrait, v, Δ, changed) = after_Δnout(base(t), v, Δ, changed)
function after_Δnout(t::SizeChangeValidation, v, Δ, changed) 
    after_Δnout(base(t), v, Δ, changed)
    validate_Δnout(v, Δ)
end
function after_Δnout(t::SizeChangeLogger, v, Δ, changed)
    if changed
        @logmsg t.level "Change nout of $(infostr(t, v)) by $(compressed_string(Δ))"
    end
    after_Δnout(base(t), v, Δ, changed)
end
function after_Δnout(t, v, Δ, changed) end

function validate_Δnin(v::AbstractVertex, Δ)
    length(Δ) == length(inputs(v)) || throw(ArgumentError("Length of Δ must be equal to number of inputs for $(nameorrepr(v))! length(Δ) = $(length(Δ)), length(inputs(v)) = $(length(inputs(v)))"))
    nout.(inputs(v)) == nin(v) || throw(ΔSizeFailError("Nin change of $(compressed_string.(Δ)) to $(nameorrepr(v)) did not result in expected size! Expected: $(nout.(inputs(v))), actual: $(nin(v))")) 
end

function validate_Δnout(v::AbstractVertex, Δ)
    nin_of_outputs = unique(mapreduce(vi -> nin(vi)[inputs(vi) .== v], vcat, outputs(v), init=nout(v)))
    nin_of_outputs == [nout(v)] || throw(ΔSizeFailError("Nout change of $(compressed_string(Δ)) to $(nameorrepr(v)) resulted in size mismatch! Nin of outputs: $nin_of_outputs, nout of this: $([nout(v)])"))
end

newsizes(s::ThrowΔSizeFailError, vs::AbstractVector{<:AbstractVertex}) = throw(ΔSizeFailError(s.msgfun(vs)))
newsizes(::ΔSizeFailNoOp, vs::AbstractVector{<:AbstractVertex}) = false, Dict(v => nin(v) for v in vs), nout.(vs)
function newsizes(s::LogΔSizeExec, vertices::AbstractVector{<:AbstractVertex})
    @logmsg s.level s.msgfun(vertices[1])
    return newsizes(base(s), vertices)
end

"""
    newsizes(s::AbstractΔSizeStrategy, vertices::AbstractArray{<:AbstractVertex})

Return a vector of new outputs sizes for and a `Dict` of new input sizes for all provided `vertices` using the strategy `s`.

Result vector is index aligned with `vertices`.
Result `Dict` has a vector of input sizes for each element of `vertices` which has an input (i.e everything except input vertices).
"""
function newsizes(s::AbstractJuMPΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex})
    case = ScalarSize()
    model = sizemodel(s, vertices)

    noutvars = @variable(model, noutvars[i=1:length(vertices)], Int)

    noutdict = Dict(zip(vertices, noutvars))
    for v in vertices
        vertexconstraints!(case, v, s, (;model, noutdict))
    end

    sizeobjective!(case, s, vertices, (;model, noutdict))

    JuMP.optimize!(model)

    if accept(case, s, model)
        return true, ninsandnouts(s, vertices, noutvars)...
    end
    return newsizes(fallback(s), vertices)
end

"""
    sizemodel(s::AbstractJuMPΔSizeStrategy, vertices)

Return a `JuMP.Model` for executing strategy `s` on `vertices`.
"""
sizemodel(::AbstractJuMPΔSizeStrategy, vertices) = JuMP.Model(JuMP.optimizer_with_attributes(Cbc.Optimizer, "loglevel"=>0))
function sizemodel(s::TimeLimitΔSizeStrategy, vs) 
    model = sizemodel(base(s), vs)
    JuMP.set_time_limit_sec(model, s.limit)
    return model
end

"""
    accept(case, ::AbstractJuMPΔSizeStrategy, model::JuMP.Model)

Return true of the solution for `model` is accepted using strategy `s`.
"""
accept(case, s::DecoratingJuMPΔSizeStrategy, model::JuMP.Model) = accept(case, base(s), model)
accept(case, ::AbstractJuMPΔSizeStrategy, model::JuMP.Model) = JuMP.termination_status(model) != MOI.INFEASIBLE && JuMP.primal_status(model) == MOI.FEASIBLE_POINT # Beware: primal_status seems unreliable for Cbc. See MathOptInterface issue #822
function accept(case, s::TimeOutAction, model::JuMP.Model)
    if  JuMP.termination_status(model) == MOI.TIME_LIMIT
        return s.action(model)
    end
    return accept(case, base(s), model)
end

# Just a shortcut for broadcasting on dicts
getall(d::Dict, ks, deffun=() -> missing) = get.(deffun, [d], ks)

# First we dispatch on trait in order to filter out immutable vertices
# First dispatch on traits to sort out things like immutable vertices
vertexconstraints!(case, v::AbstractVertex, s::AbstractJuMPΔSizeStrategy, data) = vertexconstraints!(case, trait(v), v, s, data)
vertexconstraints!(case, t::DecoratingTrait, v, s::AbstractJuMPΔSizeStrategy, data) = vertexconstraints!(case, base(t), v, s, data)
vertexconstraints!(case, ::MutationSizeTrait, v, s::AbstractJuMPΔSizeStrategy, data) = vertexconstraints!(case, s, v, data) # Now dispatch on strategy

function vertexconstraints!(case::ScalarSize, ::Immutable, v, s, data)
    @constraint(data.model, data.noutdict[v] == nout(v))
    @constraint(data.model, getall(data.noutdict, inputs(v)) .== nin(v))
    vertexconstraints!(case, s, v, data)
end

# This must be applied to immutable vertices as well
function vertexconstraints!(currcase::ScalarSize, v::AbstractVertex, s::AlignNinToNout, data, case=currcase)
    vertexconstraints!(case, v, s.vstrat, data)
    # Code below secretly assumes vo is in data.noutdict (ninarr will be left with undef entries otherwise).
    # ninvar shall be used to set new nin sizes after problem is solved
    for vo in filter(vo -> vo in keys(data.noutdict), outputs(v))
        ninvar = @variable(data.model, integer=true)
        @constraint(data.model, data.noutdict[v] == ninvar)

        ninarr = get!(() -> Vector{JuMP.VariableRef}(undef, length(inputs(vo))), s.nindict, vo)
        ninarr[inputs(vo) .== v] .= ninvar
    end
end

function vertexconstraints!(currcase::ScalarSize, v::AbstractVertex, s::AlignNinToNoutVertices, data, case=currcase)
    # Any s.ininds which are mapped to vertices which are not yet added in s.vstrat.nindict[s.vout] will cause undefined reference below
    # There is a check to wait for them to be added, but if they are not in data.noutdict they will never be added, so we need to check for that too. Not 100% this can even happen, so I'm just waiting for this in itself to trigger a bug. Sigh...
    neededinds = filter(i -> i !== nothing, indexin(keys(data.noutdict), inputs(s.vout)))

    condition() = s.vout in keys(s.vstrat.nindict) && all(i -> isassigned(s.vstrat.nindict[s.vout], i), neededinds)

    # Just to make sure we only do this once without having to store hasadded as a field in AlignNinToNoutVertices:
    #  - Check the condition before vertexconstraints! -> Only when condition changes from false to true do we add as we assume this is the first time the condition becomes fulfilled
    hasadded = condition()
    vertexconstraints!(case, v, s.vstrat, data)

    if !hasadded && condition()
        @constraint(data.model, data.noutdict[s.vin] .== s.vstrat.nindict[s.vout][s.ininds])
    end
end


"""
    vertexconstraints!(s::AbstractJuMPΔSizeStrategy, v, data)

Add constraints for `AbstractVertex v` using strategy `s`.

Extra info like the model and variables is provided in `data`.
"""
function vertexconstraints!(case::ScalarSize, s::AbstractJuMPΔSizeStrategy, v, data)
    ninconstraint!(case, s, v, data)
    compconstraint!(case, s, v, (data..., vertex=v))
    sizeconstraint!(case, s, v, data)
end

"""
    sizeconstraint!(s::AbstractJuMPΔSizeStrategy, v, data)

Add size constraints for `AbstractVertex v` using strategy `s`.

Extra info like the model and variables is provided in `data`.
"""
function sizeconstraint!(::ScalarSize, s::AbstractJuMPΔSizeStrategy, v, data) 
    # If we find any outputs to v which are not part of the problem to solve, we need to prevent v from changing size
    if any(vo -> vo ∉ keys(data.noutdict), outputs(v))
        # Ok, this is a bit speculative, but should we find that there is a size mismatch (e.g. when create/remove vertex/edge)
        # we might make the problem infeasible somehow by doing this. Instead we just skip it and hope some other constraint deals
        # with aligning the size 
        allnins = unique!(mapreduce(vo -> nin(vo)[v .== inputs(vo)], vcat, outputs(v)))
        if length(allnins) == 1
            @constraint(data.model, data.noutdict[v] == allnins[1])
            return
        end
    end
    @constraint(data.model, data.noutdict[v] >= 1)
end

function sizeconstraint!(case::ScalarSize, s::ΔNout{Exact}, v, data)
    Δ = get(s.Δs, v, nothing)
    if Δ !== nothing
        @constraint(data.model, data.noutdict[v] == nout(v) + Δ)
    else
        sizeconstraint!(case, DefaultJuMPΔSizeStrategy(), v, data)
    end
end

"""
    ninconstraint!(case, s, v, data)

Add input size constraints for `AbstractVertex v` using strategy `s`.

Extra info like the model and variables is provided in `data`.
"""
ninconstraint!(case, s, v, data) = ninconstraint!(case, s, trait(v), v, data)
ninconstraint!(case, s, t::DecoratingTrait, v, data) = ninconstraint!(case, s, base(t), v, data)
function ninconstraint!(case, s, ::MutationTrait, v, data) end
ninconstraint!(case, s, ::SizeStack, v, data) = @constraint(data.model, sum(getall(data.noutdict, inputs(v))) == data.noutdict[v])
ninconstraint!(case, s, ::SizeInvariant, v, data) = @constraint(data.model, getall(data.noutdict, unique(inputs(v))) .== data.noutdict[v])

"""
    compconstraint!(case, s, v, data)

Add constraints on the computation (e.g. neural network layer) for `AbstractVertex v` using strategy `s`.

Extra info like the model and variables is provided in `data`.
"""
compconstraint!(case, s, v::AbstractVertex, data) = compconstraint!(case, s, base(v), data)
compconstraint!(case, s, v::CompVertex, data) = compconstraint!(case, s, v.computation, data)
function compconstraint!(case, s, v::InputVertex, data) end
function compconstraint!(case, s, f, data) end


"""
    sizeobjective!(case, s::AbstractJuMPΔSizeStrategy, model, noutvars, sizetargets)

Add the objective for `noutvars` using strategy `s`.
"""
sizeobjective!(case::ScalarSize, s::AbstractJuMPΔSizeStrategy, vertices, data) = @objective(data.model, Min, objective!(case, s, vertices, data))

function objective!(::ScalarSize, s, vertices, data)
    model = data.model
    noutvars = map(v -> data.noutdict[v], vertices)
    sizetargets = nout.(vertices)
    # L1 norm prevents change in vertices which does not need to change.
    # Max norm tries to spread out the change so no single vertex takes most of the change.
    return norm!(SumNorm(0.1 => L1NormLinear(), 0.8 => MaxNormLinear()), model, @expression(model, objective[i=1:length(noutvars)], noutvars[i] - sizetargets[i]), sizetargets)
end

objective!(case::ScalarSize, s::ΔNout{Relaxed}, vertices, data) = noutrelax!(case, s.Δs, vertices, data)

function noutrelax!(case, Δs, vertices, data)
    def_obj = objective!(case, DefaultJuMPΔSizeStrategy(), setdiff(vertices, keys(Δs)), data)

    model = data.model
    # Force it to change as Δ might be too small
    # Trick from http://lpsolve.sourceforge.net/5.1/absolute.htm
    Δnout_const = @expression(model, [data.noutdict[v] - nout(v) for v in keys(Δs)])

    B = @variable(model, [1:length(Δs)], Bin)
    M = 1e5
    ϵ = 1e-2 # abs(Δnout_const) must be larger than this
    @constraint(model, Δnout_const .+ M .* B .>= ϵ)
    @constraint(model, Δnout_const .+ M .* B .<= M .- ϵ)
   
    sizediff = @expression(model, [data.noutdict[v] - nout(v) - Δ for (v, Δ) in Δs])
    Δnout_obj = norm!(SumNorm(0.1 => L1NormLinear(), 0.8 => MaxNormLinear()), model, sizediff)

    return @expression(model, def_obj + 1e6*sum(Δnout_obj))
end

function ninsandnouts(::AbstractJuMPΔSizeStrategy, vs, noutvars)
    nouts = round.(Int, JuMP.value.(noutvars))
    mapnout(i::Integer) = nouts[i]
    mapnout(i::Nothing) = missing

    nins = Dict(vs .=> map(vi -> mapnout.(indexin(inputs(vi), vs)), vs))
    return nins, nouts
end

function ninsandnouts(s::AlignNinToNout, vs, noutvars)
    nouts = round.(Int, JuMP.value.(noutvars))
    nins = Dict(key => round.(Int, JuMP.value.(value)) for (key, value) in s.nindict)
    return nins,nouts
end

ninsandnouts(s::AlignNinToNoutVertices, vs, noutvars) = ninsandnouts(s.vstrat, vs, noutvars)

# TODO: Remove since only used for debugging. If only it wasn't so bloody cumbersome to just list the constraints in a JuMP model....
nconstraints(model) = mapreduce(tt -> JuMP.num_constraints.(model,tt...), +,  filter(tt -> tt != (JuMP.VariableRef, MOI.Integer), JuMP.list_of_constraint_types(model)), init=0)

# TODO: Remove since only used for debugging. If only it wasn't so bloody cumbersome to just list the constraints in a JuMP model....
function list_constraints(model)
    for tt in filter(tt -> tt != (JuMP.VariableRef, MOI.Integer), JuMP.list_of_constraint_types(model))
        display(JuMP.all_constraints(model, tt...))
    end
end
