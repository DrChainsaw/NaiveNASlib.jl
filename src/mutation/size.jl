
#TODO: Ugh, this is too many abstraction layers for too little benefit. Refactor so
# all MutationVertex has state?
nin(t, ::InputSizeVertex) = []
nin(v::AbstractVertex) = nin(trait(v), v)
nin(t::DecoratingTrait, v::AbstractVertex) = nin(base(t), v)
nin(t::SizeAbsorb, v::AbstractVertex) = nin(t, base(v))
nin(::SizeAbsorb, v::CompVertex) = nin(v.computation)

# SizeTransparent does not need mutation state to keep track of sizes
nin(::SizeTransparent, v::AbstractVertex) = nout.(inputs(v))

nout(v::AbstractVertex) = nout(trait(v), v)
nout(t::DecoratingTrait, v::AbstractVertex) = nout(base(t), v)
nout(t::SizeAbsorb, v::AbstractVertex) = nout(t, base(v))
nout(::SizeAbsorb, v::CompVertex) = nout(v.computation)
nout(t, v::InputSizeVertex) = v.size

# SizeTransparent might not care about size
nout(::SizeInvariant, v::AbstractVertex) = nin(v)[1]
nout(::SizeStack, v::AbstractVertex) = sum(nin(v))

# TODO: Remove
nout_org(v::AbstractVertex) = nout(v)
nin_org(v::AbstractVertex) = nin(v)



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
    Δsize(case, s::AbstractΔSizeStrategy, vertices::AbstractArray{<:AbstractVertex})

Calculate new sizes for (potentially) all provided `vertices` using the strategy `s` and apply all changes.
"""
function Δsize(::ScalarSize, s::AbstractΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex})
    execute, nins, nouts = newsizes(s, vertices)
    if execute
        Δsize(nins, nouts, vertices)
    end
end

"""
    Δsize(nins::AbstractDict, nouts::AbstractVector{<:Integer}, vertices::AbstractVector{<:AbstractVertex})

Set output size of `vertices[i]` to `nouts[i]` for all `i` in `1:length(vertices)`.
Set input size of all keys `vi` in `nins` to `nins[vi]`.
"""
function Δsize(nins::AbstractDict, nouts::AbstractVector{<:Integer}, vertices::AbstractVector{<:AbstractVertex})
    Δnouts = nouts .- nout.(vertices)

    for (i, vi) in enumerate(vertices)
        ninΔs = get(() -> nin(vi), nins, vi) .- nin(vi)
        Δnin(OnlyFor(), vi, ninΔs...)
        Δnout(OnlyFor(), vi, Δnouts[i])
    end

    for (i, vi) in enumerate(vertices)
        ninΔs = get(() -> nin(vi), nins, vi) .- nin(vi)
        after_Δnin(vi, ninΔs...)
        after_Δnout(vi, Δnouts[i])
    end
end

function Δnin(s::OnlyFor, v, Δs::Maybe{<:Integer}...)
    any(skipmissing(Δs) .!= 0) || return
    Δnin(s, trait(v), v, Δs)
end
Δnin(s::AbstractΔSizeStrategy, t::DecoratingTrait, v, Δs) = Δnin(s, base(t), v, Δs)
function Δnin(s::AbstractΔSizeStrategy, t::SizeChangeLogger, v, Δs)

    @logmsg t.level "Change nin of $(infostr(t, v)) by $(join(compressed_string.(Δs), ", "))"
    Δnin(s, base(t), v, Δs)
end
Δnin(::OnlyFor, ::MutationSizeTrait, v, Δs) = Δnin(op(v), Δs...)

function Δnout(s::OnlyFor, v, Δ::Integer)
    Δ == 0 && return
    Δnout(s, trait(v), v, Δ)
end
Δnout(s::AbstractΔSizeStrategy, t::DecoratingTrait, v, Δ) = Δnout(s, base(t), v, Δ)
function Δnout(s::AbstractΔSizeStrategy, t::SizeChangeLogger, v, Δ)
    @logmsg t.level "Change nout of $(infostr(t, v)) by $(compressed_string(Δ))"
    Δnout(s, base(t), v, Δ)
end
Δnout(::OnlyFor, ::MutationSizeTrait, v, Δ) = Δnout(op(v), Δ)


after_Δnin(v, Δs...) = after_Δnin(trait(v), v, Δs)
after_Δnin(t::DecoratingTrait, v, Δs) = after_Δnin(base(t), v, Δs)
after_Δnin(t::SizeChangeValidation, v, Δs) = validate_Δnin(v, Δs, () -> after_Δnin(base(t), v, Δs))
function after_Δnin(t, v, Δs) end

after_Δnout(v, Δ) = after_Δnout(trait(v), v, Δ)
after_Δnout(t::DecoratingTrait, v, Δ) = after_Δnout(base(t), v, Δ)
after_Δnout(t::SizeChangeValidation, v, Δ) = validate_Δnout(v, Δ, () -> after_Δnout(base(t), v, Δ))
function after_Δnout(t, v, Δ) end


sizeΔ(Δ::Integer) = Δ
sizeΔ(Δ::AbstractArray) = length(Δ)
function validate_Δnin(v::AbstractVertex, Δ, Δfun, validvisit = true)

    # Yeah, this is checking more than one thing. Cba to have three different structs and methods for validation
    length(Δ) == length(inputs(v)) || throw(ArgumentError("Length of Δ must be equal to number of inputs for $(v)! length(Δ) = $(length(Δ)), length(inputs(v)) = $(length(inputs(v)))"))



    if validvisit
        # TODO base(v) makes this a bit weaker than I would have wanted. Right now it is only because testcases use smaller factors to trigger SizeStack to do unusual stuff
        Δninfactor = minΔninfactor_only_for(base(v))
        any(Δi -> sizeΔ(Δi) % Δninfactor != 0, skipmissing(Δ)) && throw(ArgumentError("Nin change of $Δ to $v is not an integer multiple of $(Δninfactor)!"))
    end

    Δfun()

    if validvisit
        nout.(inputs(v)) == nin(v) || throw(ArgumentError("Nin change of $Δ to $v did not result in expected size! Expected: $(nout.(inputs(v))), actual: $(nin(v))"))
    end
end

function validate_Δnout(v::AbstractVertex, Δ, Δfun, validvisit=true)

    if validvisit
        # TODO base(v) makes this a bit weaker than I would have wanted. Right now it is only because testcases use smaller factors to trigger SizeStack to do unusual stuff
        Δnoutfactor = minΔnoutfactor_only_for(base(v))
        sizeΔ(Δ) % Δnoutfactor != 0 && throw(ArgumentError("Nout change of $Δ to $v is not an integer multiple of $(Δnoutfactor)!"))
    end

    Δfun()

    if validvisit
        nin_of_outputs = unique(mapreduce(vi -> nin(vi)[inputs(vi) .== v], vcat, outputs(v), init=nout(v)))

        nin_of_outputs == [nout(v)] || throw(ArgumentError("Nout change of $Δ to $v resulted in size mismatch! Nin of outputs: $nin_of_outputs, nout of this: $([nout(v)])"))
    end
end


newsizes(s::ΔSizeFailError, vertices::AbstractVector{<:AbstractVertex}) = error(s.msg)
newsizes(s::ΔSizeFailNoOp, vertices::AbstractVector{<:AbstractVertex}) = false, Dict(vertices .=> nin.(vertices)), nout.(vertices)
function newsizes(s::LogΔSizeExec, vertices::AbstractVector{<:AbstractVertex})
    @logmsg s.level s.msgfun(vertices[1])
    return newsizes(s.andthen, vertices)
end

"""
    newsizes(s::AbstractΔSizeStrategy, vertices::AbstractArray{<:AbstractVertex})

Return a vector of new outputs sizes for and a `Dict` of new input sizes for all provided `vertices` using the strategy `s`.

Result vector is index aligned with `vertices`.
Result `Dict` has a vector of input sizes for each element of `vertices` which has an input (i.e everything except input vertices).
"""
function newsizes(s::AbstractJuMPΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex})

    model = sizemodel(s, vertices)

    noutvars = @variable(model, noutvars[i=1:length(vertices)], Int)

    noutdict = Dict(zip(vertices, noutvars))
    for v in vertices
        vertexconstraints!(ScalarSize(), v, s, (model=model, noutdict=noutdict))
    end

    sizeobjective!(s, model, noutvars, vertices)

    JuMP.optimize!(model)

    if accept(s, model)
        return true, ninsAndNouts(s, vertices, noutvars)...
    end
    return newsizes(fallback(s), vertices)
end

"""
    sizemodel(s::AbstractJuMPΔSizeStrategy, vertices)

Return a `JuMP.Model` for executing strategy `s` on `vertices`.
"""
sizemodel(s::AbstractJuMPΔSizeStrategy, vertices) = JuMP.Model(JuMP.optimizer_with_attributes(Cbc.Optimizer, "loglevel"=>0))

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
    vertexconstraints!(currcase, v, s.vstrat, data)
    # Code below secretly assumes vo is in data.noutdict (ninarr will be left with undef entries otherwise).
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
function vertexconstraints!(case, s::AbstractJuMPΔSizeStrategy, v, data)
    ninconstraint!(s, v, data)
    compconstraint!(case, s, v, (data..., vertex=v))
    sizeconstraint!(case, s, v, data)
end

"""
    sizeconstraint!(s::AbstractJuMPΔSizeStrategy, v, data)

Add size constraints for `AbstractVertex v` using strategy `s`.

Extra info like the model and variables is provided in `data`.
"""
sizeconstraint!(::ScalarSize, s::AbstractJuMPΔSizeStrategy, v, data) = @constraint(data.model, data.noutdict[v] >= 1)

function sizeconstraint!(case::ScalarSize, s::ΔNout{Exact}, v, data)
    if v == s.vertex
        @constraint(data.model, data.noutdict[v] == nout(v) + s.Δ)
    else
        sizeconstraint!(case, DefaultJuMPΔSizeStrategy(), v, data)
    end
end

function sizeconstraint!(case::ScalarSize, s::ΔNin{Exact}, v, data)
    if v in s.vertices
        Δ = s.Δs[s.vertices .== v][1]
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
ninconstraint!(s, v, data) = ninconstraint!(s, trait(v), v, data)
ninconstraint!(s, t::DecoratingTrait, v, data) = ninconstraint!(s, base(t), v, data)
function ninconstraint!(s, ::MutationTrait, v, data) end
ninconstraint!(s, ::SizeStack, v, data) = @constraint(data.model, sum(getall(data.noutdict, inputs(v))) == data.noutdict[v])
ninconstraint!(s, ::SizeInvariant, v, data) = @constraint(data.model, getall(data.noutdict, unique(inputs(v))) .== data.noutdict[v])



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
sizeobjective!(s::AbstractJuMPΔSizeStrategy, model, noutvars, vertices) = @objective(model, Min, objective!(ScalarSize(), s, model, noutvars, vertices))

function objective!(::ScalarSize, s, model, noutvars, vertices)
    sizetargets = nout.(vertices)
    # L1 norm prevents change in vertices which does not need to change.
    # Max norm tries to spread out the change so no single vertex takes most of the change.
    return norm!(SumNorm(0.1 => L1NormLinear(), 0.8 => MaxNormLinear()), model, @expression(model, objective[i=1:length(noutvars)], noutvars[i] - sizetargets[i]), sizetargets)
end

objective!(s::ΔNout{Relaxed}, model, noutvars, vertices) = noutrelax!(model, [s.vertex], [s.Δ], noutvars, vertices)
objective!(s::ΔNin{Relaxed}, model, noutvars, vertices) = noutrelax!(model, s.vertices, s.Δs, noutvars, vertices)

function noutrelax!(model, vs, Δs, noutvars, vertices)
    inds = mapreduce(v -> vertices .== v, (i1,i2) -> i1 .| i2, vs)
    def_obj = objective!(DefaultJuMPΔSizeStrategy(), model, noutvars[.!inds], vertices[.!inds])
    sizetarget = nout.(vs) + Δs
    Δnout_obj = norm!(L1NormLinear(), model, @expression(model, noutvars[inds] .- sizetarget))
    # Force it to change as s.Δ might be too small
    # Trick from http://lpsolve.sourceforge.net/5.1/absolute.htm
    Δnout_const = @expression(model, noutvars[inds] - nout.(vs))
    B = @variable(model, [1:length(vs)], binary=true)
    M = 1e5
    ϵ = 1e-2 # abs(Δnout_const) must be larger than this
    @constraint(model, Δnout_const .+ M .* B .>= ϵ)
    @constraint(model, Δnout_const .+ M .* B .<= M .- ϵ)

    return @expression(model, def_obj + 1e6*sum(Δnout_obj))
end

"""
    accept(::AbstractJuMPΔSizeStrategy, model::JuMP.Model)

Return true of the solution for `model` is accepted using strategy `s`.
"""
accept(::AbstractJuMPΔSizeStrategy, model::JuMP.Model) = JuMP.termination_status(model) != MOI.INFEASIBLE && JuMP.primal_status(model) == MOI.FEASIBLE_POINT

function ninsAndNouts(::AbstractJuMPΔSizeStrategy, vs, noutvars)
    nouts = round.(Int, JuMP.value.(noutvars))
    mapnout(i::Integer) = nouts[i]
    mapnout(i::Nothing) = missing

    nins = Dict(vs .=> map(vi -> mapnout.(indexin(inputs(vi), vs)), vs))
    return nins, nouts
end

function ninsAndNouts(s::AlignNinToNout, vs, noutvars)
    nouts = round.(Int, JuMP.value.(noutvars))
    nins = Dict(key => round.(Int, JuMP.value.(value)) for (key, value) in s.nindict)
    return nins,nouts
end

ninsAndNouts(s::AlignNinToNoutVertices, vs, noutvars) = ninsAndNouts(s.vstrat, vs, noutvars)

# TODO: Remove since only used for debugging. If only it wasn't so bloody cumbersome to just list the constraints in a JuMP model....
nconstraints(model) = mapreduce(tt -> JuMP.num_constraints.(model,tt...), +,  filter(tt -> tt != (JuMP.VariableRef, MOI.Integer), JuMP.list_of_constraint_types(model)), init=0)

# TODO: Remove since only used for debugging. If only it wasn't so bloody cumbersome to just list the constraints in a JuMP model....
function list_constraints(model)
    for tt in filter(tt -> tt != (JuMP.VariableRef, MOI.Integer), JuMP.list_of_constraint_types(model))
        display(JuMP.all_constraints(model, tt...))
    end
end
