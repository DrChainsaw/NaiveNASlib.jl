

# Vertex traits w.r.t whether size changes propagates
"""
    SizeTransparent
Base type for mutation traits which are transparent w.r.t size, i.e size changes propagate both forwards and backwards.
"""
abstract type SizeTransparent <: MutationSizeTrait end
"""
    SizeStack
Transparent size trait type where inputs are stacked, i.e output size is the sum of all input sizes
"""
struct SizeStack <: SizeTransparent end
"""
    SizeInvariant
Transparent size trait type where all input sizes must be equal to the output size, e.g. elementwise operations (including broadcasted).
"""
struct SizeInvariant <: SizeTransparent end
"""
    SizeAbsorb
Size trait type for which size changes are absorbed, i.e they do not propagate forward.

Note that size changes do propagate backward as changing the input size of a vertex requires that the output size of its input is also changed and vice versa.
"""
struct SizeAbsorb <: MutationSizeTrait end

"""
    VisitState

Memoization struct for traversal.

Remembers visitation for both forward (in) and backward (out) directions.
"""
mutable struct VisitState{T}
    in::Vector{AbstractVertex}
    out::Vector{AbstractVertex}
    ninΔs::OrderedDict{AbstractVertex, Vector{Maybe{<:T}}}
    noutΔs::OrderedDict{AbstractVertex, Maybe{<:T}}
    change_nin::Dict{AbstractVertex, Vector{Bool}}
end
VisitState{T}() where T = VisitState{T}(Dict{AbstractVertex, Vector{Bool}}())
VisitState{T}(change_nin::Dict{AbstractVertex, Vector{Bool}}) where T = VisitState{T}([], [], OrderedDict{MutationVertex, Vector{Maybe{<:T}}}(), OrderedDict{MutationVertex, Maybe{<:T}}(), change_nin)
VisitState{T}(origin::AbstractVertex) where T  = VisitState{T}(Δnout_touches_nin(origin).touch_nin)
VisitState{T}(origin::AbstractVertex, Δs::Maybe{T}...) where T  = VisitState{T}(Δnin_touches_nin(origin).touch_nin) # Here it *should* be required to send Δs to Δnin_touches_nin so that missing Δs can be masked, but so far it has not turned out to be neccessary...

Base.Broadcast.broadcastable(s::VisitState) = Ref(s)

visited_in!(s::VisitState, v::AbstractVertex) = push!(s.in, v)
visited_out!(s::VisitState, v::AbstractVertex) = push!(s.out, v)
has_visited_in(s::VisitState, v::AbstractVertex) = v in s.in
has_visited_out(s::VisitState, v::AbstractVertex) = v in s.out

ninΔs(s::VisitState{T}) where T = s.ninΔs
noutΔs(s::VisitState{T}) where T = s.noutΔs

# Only so it is possible to broadcast since broadcasting over dics is reserved
getnoutΔ(defaultfun, s::VisitState{T}, v::AbstractVertex) where T = get(defaultfun, s.noutΔs, v)
setnoutΔ!(Δ::T, s::VisitState{T}, v::AbstractVertex) where T = s.noutΔs[v] = Δ
function setnoutΔ!(missing, s::VisitState, v::AbstractVertex) end


#TODO: Ugh, this is too many abstraction layers for too little benefit. Refactor so
# all MutationVertex has state?
nin(v::InputSizeVertex) = v.size
nin(v::MutationVertex) = nin(v, op(v))
nin(v::AbstractVertex, op::MutationState) = nin(op)
nin(v::AbstractVertex, op::MutationOp) = nin(trait(v), v)
nin(t::DecoratingTrait, v::AbstractVertex) = nin(base(t), v)

# SizeTransparent does not need mutation state to keep track of sizes
nin(::SizeTransparent, v::AbstractVertex) = nout.(inputs(v))

nout(v::InputSizeVertex) = v.size
nout(v::MutationVertex) = nout(v, op(v))
nout(v::AbstractVertex, op::MutationState) = nout(op)
nout(v::AbstractVertex, op::MutationOp) = nout(trait(v), v)
nout(t::DecoratingTrait, v::AbstractVertex) = nout(base(t), v)

# SizeTransparent does not need mutation state to keep track of sizes
nout(t::SizeInvariant, v::AbstractVertex) = nin(v)[1]
nout(t::SizeStack, v::AbstractVertex) = sum(nin(v))

nout_org(v::AbstractVertex) = nout_org(trait(v), v)
nout_org(t::DecoratingTrait, v) = nout_org(base(t), v)
nout_org(::MutationSizeTrait, v::MutationVertex) = nout_org(op(v))
nout_org(::Immutable, v) = nout(v)

nin_org(v::AbstractVertex) = nin_org(trait(v), v)
nin_org(t::DecoratingTrait, v) = nin_org(base(t), v)
nin_org(::MutationSizeTrait, v::MutationVertex) = nin_org(op(v))
nin_org(::Immutable, v) = nout(v)


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
minΔnoutfactor_only_for(v::AbstractVertex, s=VisitState{Int}()) = outvisit(v, s) ? 1 : minΔnoutfactor_only_for(base(v),s)
minΔnoutfactor_only_for(v::MutationVertex,s=VisitState{Int}()) = outvisit(v, s) ? 1 : minΔnoutfactor_only_for(trait(v), v, s)
minΔnoutfactor_only_for(v::InputVertex,s=VisitState{Int}())= missing
minΔnoutfactor_only_for(v::CompVertex,s=VisitState{Int}()) = minΔnoutfactor(v.computation)
minΔnoutfactor(f::Function) = 1 # TODO: Move to test as this does not make alot of sense
minΔnoutfactor_only_for(t::DecoratingTrait, v::AbstractVertex, s) = minΔnoutfactor_only_for(base(t), v, s)
minΔnoutfactor_only_for(::Immutable, v::AbstractVertex, s) = missing
minΔnoutfactor_only_for(::SizeAbsorb, v::AbstractVertex, s) = minΔnoutfactor_only_for(base(v),s)

minΔnoutfactor_only_for(::SizeInvariant, v::AbstractVertex, s) = lcmsafe(vcat(minΔnoutfactor_only_for.(inputs(v),s), minΔninfactor_only_for.(outputs(v),s)))
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
minΔninfactor_only_for(v::AbstractVertex, s=VisitState{Int}()) = invisit(v, s) ? 1 : minΔninfactor_only_for(base(v),s)
minΔninfactor_only_for(v::MutationVertex,s=VisitState{Int}()) = invisit(v, s) ? 1 : minΔninfactor_only_for(trait(v), v, s)
minΔninfactor_only_for(v::InputVertex,s=VisitState{Int}())  = missing
minΔninfactor_only_for(v::CompVertex,s=VisitState{Int}()) = minΔninfactor(v.computation)
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


## Boilerplate

# Dispatch on trait
Δnin(v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}(v, Δ...)) where T = Δnin(trait(v), v, Δ..., s=s)
Δnout(v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}(v)) where T = Δnout(trait(v), v, Δ, s=s)

# Unwrap DecoratingTrait(s)
Δnin(t::DecoratingTrait, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T = Δnin(base(t), v, Δ..., s=s)
Δnout(t::DecoratingTrait, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T = Δnout(base(t), v, Δ, s=s)

# Potential failure case: Try to change immutable vertex
Δnin(::Immutable, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T = !has_visited_in(s, v) && any(skipmissing(Δ) .!= 0) && error("Tried to change nin of immutable $v to $Δ")
Δnout(::Immutable, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T = !has_visited_out(s, v) && Δ != 0 && error("Tried to change nout of immutable $v to $Δ")

# Logging
function Δnin(t::SizeChangeLogger, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T

    !has_visited_in(s, v) && @logmsg t.level "Change nin of $(infostr(t, v)) by $Δ"
    Δnin(base(t), v, Δ..., s=s)
end

function Δnout(t::SizeChangeLogger, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T

    !has_visited_out(s, v) && @logmsg t.level "Change nout of $(infostr(t, v)) by $Δ"
    Δnout(base(t), v, Δ, s=s)
end

# Validation
sizeΔ(Δ::Integer) = Δ
sizeΔ(Δ::AbstractArray) = length(Δ)
function Δnin(t::SizeChangeValidation, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T

    # Yeah, this is checking more than one thing. Cba to have three different structs and methods for validation
    length(Δ) == length(inputs(v)) || throw(ArgumentError("Length of Δ must be equal to number of inputs for $(v)! length(Δ) = $(length(Δ)), length(inputs(v)) = $(length(inputs(v)))"))

    validvisit = !has_visited_in(s, v)

    if validvisit
        # TODO base(v) makes this a bit weaker than I would have wanted. Right now it is only because testcases use smaller factors to trigger SizeStack to do unusual stuff
        Δninfactor = minΔninfactor_only_for(base(v))
        any(Δi -> sizeΔ(Δi) % Δninfactor != 0, skipmissing(Δ)) && throw(ArgumentError("Nin change of $Δ to $v is not an integer multiple of $(Δninfactor)!"))
    end

    Δnin(base(t), v, Δ..., s=s)

    if validvisit
        nout.(inputs(v)) == nin(v) || throw(ArgumentError("Nin change of $Δ to $v did not result in expected size! Expected: $(nout.(inputs(v))), actual: $(nin(v))"))
    end
end


function Δnout(t::SizeChangeValidation, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    validvisit = !has_visited_out(s, v) && !(v in keys(noutΔs(s)))

    if validvisit
        # TODO base(v) makes this a bit weaker than I would have wanted. Right now it is only because testcases use smaller factors to trigger SizeStack to do unusual stuff
        Δnoutfactor = minΔnoutfactor_only_for(base(v))
        sizeΔ(Δ) % Δnoutfactor != 0 && throw(ArgumentError("Nout change of $Δ to $v is not an integer multiple of $(Δnoutfactor)!"))
    end

    Δnout(base(t), v, Δ, s=s)

    if validvisit
        nin_of_outputs = unique(mapreduce(vi -> nin(vi)[inputs(vi) .== v], vcat, outputs(v), init=nout(v)))

        nin_of_outputs == [nout(v)] || throw(ArgumentError("Nout change of $Δ to $v resulted in size mismatch! Nin of outputs: $nin_of_outputs, nout of this: $([nout(v)])"))
    end
end


# Actual operations

function Δnin(::SizeAbsorb, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T
    invisit(v, s) && return
    Δnin(op(v), Δ...)
    propagate_nout(v, Δ..., s=s)
end

function Δnout(::SizeAbsorb, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    outvisit(v,s) && return
    Δnout(op(v), Δ)
    propagate_nin(v, Δ, s=s)
end


function Δnin(::SizeStack, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return
    # Need to calculate concat value before changing nin
    Δo = concat(op(v), Δ...)

    Δnin(op(v), Δ...)
    propagate_nout(v, Δ...; s=s)

    Δnout(op(v), Δo)
    propagate_nin(v, Δo; s=s)

end

function Δnout(::SizeStack, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    Δnout(op(v), Δ)
    propagate_nin(v, Δ, s=s) # If there are multiple outputs they must all be updated

    Δs = split_nout_over_inputs(v, Δ, s)
    Δnin(op(v), Δs...)
    propagate_nout(v, Δs...; s=s)
end

concat(::MutationOp, Δ::Maybe{T}...) where T = sum(filter(!ismissing, collect(Δ)))
concat(o::MutationOp, Δ::Maybe{AbstractVector{T}}...) where T = concat(nin(o), in_inds(o), Δ...)
concat(o::IoChange, Δ::Maybe{AbstractVector{T}}...) where T = concat(nin_org(o), in_inds(o), Δ...)


function concat(insizes, currinds, Δ::Maybe{AbstractArray{T}}...) where T
    Δ = collect(Δ)
    missing_inds = ismissing.(Δ)
    Δ[missing_inds] = currinds[missing_inds]

    res = Δ[1]
    for (i, Δi) in enumerate(Δ[2:end])
        res = vcat(res, Δi + sign.(Δi) * sum(insizes[1:i]))
    end

    return res
end

function split_nout_over_inputs(v::AbstractVertex, Δ::T, s::VisitState{T}) where T<:Integer
    # All we want is basically a split of Δ weighted by each individual input size
    # Major annoyance #1: We must comply to the Δfactors so that Δi for input i is an
    # integer multiple of Δfactors[i]
    # Major annoyance #2: Size might already have been propagated to some of the vertices through another path. Need to account for that by digging deeply for non-size-transparent vertices (terminating vertices) and distribute sizes for each one of them to ensure things will work out when we get there. Stuff of nightmares!

    # Note: terminating_vertices is an array of arrays so that terminating_vertices[i] are all terminating vertices seen through input vertex i
    # We will use it later to accumulate all individual size changes in that direction

    terminating_vertices = findterminating.(inputs(v), inputs) # Need to also add outputs and put v in memo?

    #ftv = flattened_terminating_vertices, okay?
    ftv = vcat(terminating_vertices...)
    uftv = unique(ftv)

    # Find which sizes has not been determined through some other path; those are the ones we shall consider here
    termΔs::AbstractArray{Maybe{T}} = getnoutΔ.(() -> missing, s, uftv)
    missinginds = ismissing.(termΔs)

    if any(missinginds)
        #Yeah, muftv = missing_unique_flattened_terminating_vertices
        muftv = uftv[missinginds]
        # Only split parts which are not already set through some other path.
        Δ -= mapreduce(vt -> getnoutΔ(() -> 0, s, vt), +, ftv)

        # Remap any duplicated vertices Δf_i => 2 * Δf_i
        Δfactors = Integer[count(x->x==va,ftv) * minΔnoutfactor_only_for(va) for va in muftv]
        insizes = nout.(muftv)

        # Weighted split so that larger insizes take more of the change
        weights = Δfactors ./ insizes

        # Optimization: "Shave off" as many integer multiples of minΔninfactor as we can
        minΔf = minΔninfactor(v)
        Δpre = abs(Δ) <= minΔf ? 0 : (Δ ÷ minΔf - sign(Δ)) * minΔf
        presplit = round.(T, (Δpre ./ weights ./ sum(insizes))) .*Δfactors
        presplit[-presplit .>= insizes] .= 0

        weights = Δfactors ./ (insizes + presplit)
        # floor is due to assumption the minimum size is 1 * Δfactors
        limits = floor.((insizes .+ presplit) ./ Δfactors)

        objective = function(n)
            # If decreasing (Δ < 0) consider any solution which would result in a nin[i] < Δfactors[i] to be invalid
            Δ < 0 && any(n .>= limits) && return Inf
            # Minimize standard deviation of Δs weighted by input size so larger input sizes get larger Δs
            return std(n .* weights, corrected=false)
        end

        nΔfactors = pick_values(Δfactors, abs(Δ - sum(presplit)), objective)

        sum(nΔfactors .* Δfactors .+ abs.(presplit)) == abs(Δ) || @warn "Failed to distribute Δ = $Δ using Δfactors = $(Δfactors) and limits $(limits)!. Proceed with $(nΔfactors) in case receiver can work it out anyways..."

        # Remap any duplicated vertices Δi =>  Δi / 2
        scale_duplicate = Integer[count(x->x==va,ftv) for va in muftv]
        termΔs[missinginds] = div.(sign(Δ) .* nΔfactors .* Δfactors .+presplit, scale_duplicate)
    end

    # Now its time to accumulate all Δs for each terminating_vertices array. Remember that terminating_vertices[i] is an array of the terminating vertices seen through input vertex i
    vert2size = Dict(uftv .=> termΔs)

    return  map(terminating_vertices) do varr
        res = mapreduce(va -> vert2size[va], +, varr, init=0)
        return res
    end
end


function split_nout_over_inputs(v::AbstractVertex, Δ::AbstractVector{T}, s::VisitState{<:AbstractVector{T}}) where T<:Integer
    boundaries(o::MutationOp, v) = nin(v)
    boundaries(o::IoChange, v) = nin_org(o)

    bounds = boundaries(op(v), v)
    Δs = ntuple(i -> T[], length(bounds))

    negind = 1
    function push(elem::Integer, ind::Integer)
        if elem < 0
            push!(Δs[negind], elem)
        elseif elem <= bounds[ind] || ind == length(bounds)
            negind = ind
            push!(Δs[ind], elem)
        else
            push(elem - bounds[ind], ind+1)
        end
    end

    foreach(elem -> push(elem, 1), Δ)
    return Δs
end


function Δnin(::SizeInvariant, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    Δprop = [Δi for Δi in unique((Δ)) if !ismissing(Δi)]
    @assert length(Δprop) == 1 "Change must be invariant! Got Δ = $Δ"

    Δnins = repeat(Δprop, length(inputs(v)))

    Δnin(op(v), Δnins...)
    Δnout(op(v), Δprop...)

    propagate_nout(v, Δnins...; s=s)
    propagate_nin(v, Δprop...; s=s)
end

function Δnout(::SizeInvariant, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    Δnout(op(v), Δ)
    Δnin(op(v), repeat([Δ], length(inputs(v)))...)

    propagate_nin(v, Δ, s=s)
    propagate_nout(v, fill(Δ, length(inputs(v)))...; s=s)
end

## Generic helper methods

function invisit(v::AbstractVertex, s::VisitState{T}) where T
    has_visited_in(s, v) && return true
    visited_in!(s, v)
    return false
end

function outvisit(v::AbstractVertex, s::VisitState{T}) where T
    has_visited_out(s, v) && return true
    visited_out!(s, v)
    return false
end

function anyvisit(v::AbstractVertex, s::VisitState{T}) where T
    in = invisit(v, s)
    out = outvisit(v, s)
    return in || out
end

function propagate_nin(v::MutationVertex, Δ::T; s::VisitState{T}) where T
    # Rundown of the idea here: The outputs of v might have more than one input
    # If such a vertex vi is found, the missing inputs are set to "missing" and
    # the Δ we have is put in a context for vi. Only if no input is missing
    # do we propagate to vi.
    # If we end up here though another input to vi the context will be populated
    # with the new Δ and eventually we have all the Δs
    # See testset "Transparent residual fork block" and "StackingVertex multi inputs" for a motivation

    for vi in outputs(v)
        ins = inputs(vi)
        Δs = get!(ninΔs(s), vi) do
            Vector{Maybe{T}}(missing, length(ins))
        end
        # Add Δ for each input which is the current vertex (at least and typically one)
        for ind in findall(vx -> vx == v, ins)
            Δs[ind] = align_invalid_indices(vi, o -> nin_org(o)[ind], Δ)
        end

        validΔs = .!ismissing.(Δs)
        expectedΔs = get(() -> trues(length(Δs)), s.change_nin, vi)

        all(validΔs[expectedΔs]) && Δnin(vi, Δs...; s=s)
    end
end

function propagate_nout(v::MutationVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T
    Δs = align_invalid_indices.(inputs(v), nout_org, Δ)

    # Need to remember which Δs have been set, mainly when splitting a Δ over sevaral inputs as done in Δnout(::SizeStack...) so that we don't come to a different conclusion there about some vertex we have already visisted.
    setnoutΔ!.(Δs, s, inputs(v))
    for (Δi, vi) in zip(Δs, inputs(v))
        if !ismissing(Δi)
            Δnout(vi, Δi; s=s)
        end
    end
end

align_invalid_indices(v, fun, Δ) = Δ
align_invalid_indices(v::MutationVertex, fun, Δ) = align_invalid_indices(op(v), fun, Δ)
function align_invalid_indices(o::IoChange, fun, Δ::AbstractVector{T}) where T <: Integer
    # Handle cases when inputs and outputs had different sizes originally. Example of this is when a vertex between them was removed.
    maxind = fun(o)
    Δ = copy(Δ)
    Δ[Δ .> maxind] .= -1
    return Δ
end


# Yeah, this is an exhaustive search. Non-linear objective (such as std) makes
# DP not feasible (or maybe I'm just too stupid to figure out the optimal substructure).
# Shouldn't matter as any remotely sane application of this library should call this function with relatively small targets and few values
function pick_values(
    values::Vector{T},
    target::T,
    ind::T,
    n::Vector{T},
    objective) where T <: Integer

    #Base cases
    ind > length(n) && return n, Inf
    target < 0 && return n, Inf
    target == 0 && return n, objective(n)

    # Recursions
    nn = copy(n)
    nn[ind] += 1
    pick, obj_p = pick_values(values, target-values[ind], ind, nn, objective)
    next, obj_n = pick_values(values, target, ind+1, n, objective)
    return obj_p < obj_n ? (pick, obj_p) : (next, obj_n)
end

function pick_values(values::Vector{T}, target::T, objective= x->std(x, corrected=false)) where T <:Integer
    return pick_values(values, target, 1, zeros(T, length(values)), objective)[1]
end

"""
    ΔSizeInfo

Holds information on how size changes will propagate.

Field `touch_nin` is a `Dict{AbstractVertex, Vector{Bool}}` with mappings `vi => Δi` where `Δi[n] == true` if input `n` of vertex `vi` will change.
Field `touch_nout` is a `Vector{AbstractVertex}` with vertices `vi` for which nout will change.
"""
struct ΔSizeInfo
    touch_nin::Dict{AbstractVertex, Vector{Bool}}
    touch_nout::Vector{AbstractVertex}
end
ΔSizeInfo() = ΔSizeInfo(Dict{AbstractVertex, Vector{Bool}}(), AbstractVertex[])
ΔnoutSizeInfo() = Δnout_touches_nin(v, ΔSizeInfo())
ΔninSizeInfo() = Δnin_touches_nin(v, ΔSizeInfo())

update_state_nout_impl!(s::ΔSizeInfo,v,from) = push!(s.touch_nout, v)
visited_out(s::ΔSizeInfo, v) = v in s.touch_nout
clear_state_nin!(s::ΔSizeInfo, v) = delete!(s.touch_nin, v)

function update_state_nin_impl!(s::ΔSizeInfo,v,from)
    inmap = inputs(v) .== from
    notnew = v in keys(s.touch_nin) && all(s.touch_nin[v][inmap])
    stored_inmap = get!(() -> inmap, s.touch_nin, v)
    stored_inmap .|= inmap
    return notnew
end


Δnin_touches_nin(v, s=ΔSizeInfo()) = Δnin_touches_nin(trait(v), v, s)
Δnin_touches_nin(t::DecoratingTrait, v, s) = Δnin_touches_nin(base(t), v, s)
Δnin_touches_nin(::SizeAbsorb, v, s) = mapreduce(vi -> Δnout_touches_nin(vi, v, s), (s1,s2) -> s1, inputs(v), init=s)
function Δnin_touches_nin(::SizeInvariant, v, s)
    foreach(vi -> Δnout_touches_nin(vi, v, s), inputs(v))
    foreach(vo -> Δnin_touches_nin(vo, v, s), outputs(v))
    return s
end
function Δnin_touches_nin(::SizeStack, v, s)
    foreach(vi -> Δnout_touches_nin(vi, v, s), inputs(v))
    foreach(vo -> Δnin_touches_nin(vo, v, s), outputs(v))
    return s
end

function Δnin_touches_nin(v, from, s)
    update_state_nin!(s, v, from) && return s
    Δnin_touches_nin(trait(v), v, from, s)
 end
Δnin_touches_nin(::Immutable, v, from, s) = s
Δnin_touches_nin(t::DecoratingTrait, v, from, s) = Δnin_touches_nin(base(t), v, from, s)
function Δnin_touches_nin(::SizeAbsorb, v, from, s)
    foreach(vi -> Δnout_touches_nin(vi, v, s), filter(vi -> vi != from, inputs(v)))
    return s
end
function Δnin_touches_nin(::SizeInvariant, v, from, s)
    foreach(vi -> Δnout_touches_nin(vi, v, s), filter(vi -> vi != from, inputs(v)))
    foreach(vo -> Δnin_touches_nin(vo, v, s), filter(vo -> vo != from, outputs(v)))
    return s
end
function Δnin_touches_nin(::SizeStack, v, from, s)
    foreach(vo -> Δnin_touches_nin(vo, v, s), outputs(v))
    return s
end


Δnout_touches_nin(v, s=ΔSizeInfo()) = Δnout_touches_nin(trait(v), v, s)
Δnout_touches_nin(t::DecoratingTrait, v, s) = Δnout_touches_nin(base(t), v, s)
Δnout_touches_nin(::Immutable, v, s) = s
function Δnout_touches_nin(::SizeAbsorb, v, s)
    foreach(vo -> Δnin_touches_nin(vo, v, s), outputs(v))
    return s
end
Δnout_touches_nin(t::SizeTransparent, v, s) = Δnin_touches_nin(t, v, s)


function Δnout_touches_nin(v, from, s)
    update_state_nout!(s,v,from)
    Δnout_touches_nin(trait(v), v, from, s)
end
Δnout_touches_nin(t::DecoratingTrait, v, from, s) = Δnout_touches_nin(base(t), v, from, s)
Δnout_touches_nin(::Immutable, v, from, s) = s
Δnout_touches_nin(::SizeAbsorb, v, from, s) = mapreduce(vo -> Δnin_touches_nin(vo, v, s), (s1,s2) -> s1,filter(vo -> vo != from, outputs(v)), init=s)

function Δnout_touches_nin(::SizeInvariant, v, from, s)
    foreach(vi -> Δnout_touches_nin(vi, v, s), filter(vi -> vi != from, inputs(v)))
    foreach(vo -> Δnin_touches_nin(vo, v, s), filter(vo -> vo != from, outputs(v)))
    return s
end
function Δnout_touches_nin(::SizeStack, v, from, s)
    foreach(vi -> Δnout_touches_nin(vi, v, s), filter(vi -> vi != from, inputs(v)))
    foreach(vo -> Δnin_touches_nin(vo, v, s), filter(vo -> vo != from, outputs(v)))
    return s
end

update_state_nout!(s, v, from) = update_state_nout!(trait(v), s, v, from)
update_state_nout!(t::DecoratingTrait, s, v, from) = update_state_nout!(base(t), s, v, from)
update_state_nout!(::MutationTrait, s, v, from) = update_state_nout_impl!(s, v, from)
function update_state_nout!(::SizeStack,s,v,from)
    update_state_nout_impl!(s, v, from)
    clear_state_nin!(s, v) # Must resolve sizes from the nout-direction if we ever hit it
end

update_state_nin!(s, v, from) = update_state_nin!(trait(v), s, v, from)
update_state_nin!(t::DecoratingTrait, s, v, from) =  update_state_nin!(base(t), s, v, from)
update_state_nin!(::SizeAbsorb, s, v, from) = update_state_nin_impl!(s,v,from)
update_state_nin!(::Immutable, s, v, from) = true

function update_state_nin!(::SizeTransparent, s, v, from)
    visited_out(s, v) && return true
    return update_state_nin_impl!(s,v,from)
end

abstract type Direction end
struct Input <: Direction end
struct Output <: Direction end

struct ΔSizeGraph
    graph::SimpleDiGraph
    directions::Vector{Direction}
    vertices::Vector{AbstractVertex}
end

ΔninSizeGraph(v) = ΔSizeGraph(Input(), v)
ΔnoutSizeGraph(v) = ΔSizeGraph(Output(), v)

function ΔSizeGraph(::Input, v)
    g = ΔSizeGraph(SimpleDiGraph(), [], [])
    Δnin_touches_nin(v, g)
end

function ΔSizeGraph(::Output, v)
    g = ΔSizeGraph(SimpleDiGraph(), [], [])
    Δnout_touches_nin(v, g)
end

function vertexind!(g::ΔSizeGraph, v::AbstractVertex,)
    ind = indexin([v], g.vertices)[]
    if isnothing(ind)
        add_vertex!(g.graph)
        push!(g.vertices, v)
        ind = length(g.vertices)
    end
    return ind
end

function LightGraphs.add_edge!(g::ΔSizeGraph, src::AbstractVertex, dst::AbstractVertex, d::Direction)
    srcind = vertexind!(g, dst)
    dstind = vertexind!(g, src)
    has_edge(g.graph, srcind, dstind) && return true
    add_edge!(g.graph, srcind, dstind)
    push!(g.directions, d)
    return false
end

update_state_nout_impl!(g::ΔSizeGraph,v,from) = add_edge!(g, from, v, Output())
update_state_nin_impl!(g::ΔSizeGraph,v,from)  = add_edge!(g, from, v, Input())

function visited_out(g::ΔSizeGraph, v)
    inds = findall(vx -> vx == v, g.vertices)
    return any(d -> d == Output(), g.directions[inds])
end

function clear_state_nin!(g::ΔSizeGraph, v)
     v in s.vertices || return
     v_ind = vertexind(v, g)
     # No need to delete the vertex, only edges matter
     # Furthermore, this function is only called when v anyways shall be in the graph
     rem_e = []
     for (e_ind, e) in enumerate(edges(g.graph))
         if e.dst == v_ind && g.directions[e_ind] == Input()
             rem_edge!(g.graph, e)
             push!(rem_e, e_ind)
         end
     end
    deleteat!(g.directions, rem_e)
end


## Generic helper methods end
