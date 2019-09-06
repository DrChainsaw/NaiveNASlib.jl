

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
VisitState{T}(origin::AbstractVertex) where T  = VisitState{T}(ΔnoutSizeInfo(origin).touch_nin)
VisitState{T}(origin::AbstractVertex, Δs::Maybe{T}) where T  = VisitState{T}(ΔninSizeInfo(origin).touch_nin)
VisitState{T}(origin::AbstractVertex, Δs::Maybe{T}...) where T  = VisitState{T}(ΔninSizeInfo(origin, ismissing.(Δs)...).touch_nin)

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
nin(v::InputSizeVertex) = []
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

nin_org(v::InputSizeVertex) = []
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
"""
    AbstractΔSizeStrategy

Abstract base type for strategies for how to change the size.

Only used as a transition until JuMP approach has been fully verified.
"""
abstract type AbstractΔSizeStrategy end

struct ΔNoutLegacy <: AbstractΔSizeStrategy end
struct ΔNinLegacy <: AbstractΔSizeStrategy end

# TODO: Remove once new way is verified with dependent packages
global defaultΔNoutStrategy = ΔNoutLegacy()
global defaultΔNinStrategy = ΔNinLegacy()
export set_defaultΔNoutStrategy
export set_defaultΔNinStrategy
function set_defaultΔNoutStrategy(s)
    global defaultΔNoutStrategy = s
end
function set_defaultΔNinStrategy(s)
    global defaultΔNinStrategy = s
end

# Dispatch on strategy
Δnin(v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}(v, Δ...), strategy=defaultΔNinStrategy) where T = Δnin(strategy, v, Δ..., s=s)
Δnout(v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}(v), strategy=defaultΔNoutStrategy) where T = Δnout(strategy, v, Δ, s=s)

# Dispatch on trait for legacy approach (graph traversal)
Δnin(::ΔNinLegacy, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}(v, Δ...)) where T = Δnin(trait(v), v, Δ..., s=s)
Δnout(::ΔNoutLegacy, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}(v)) where T = Δnout(trait(v), v, Δ, s=s)


# Unwrap DecoratingTrait(s)
Δnin(t::DecoratingTrait, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T = Δnin(base(t), v, Δ..., s=s)
Δnout(t::DecoratingTrait, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T = Δnout(base(t), v, Δ, s=s)

# Potential failure case: Try to change immutable vertex
Δnin(::Immutable, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T = !has_visited_in(s, v) && any(skipmissing(Δ) .!= 0) && error("Tried to change nin of immutable $v to $Δ")
Δnout(::Immutable, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T = !has_visited_out(s, v) && Δ != 0 && error("Tried to change nout of immutable $v to $Δ")
NaiveNASlib.Δnout(::Immutable, v::AbstractVertex, Δ::T; s) where T<:AbstractArray{<:Integer} = !NaiveNASlib.has_visited_out(s, v) && Δ != 1:nout(v) && error("Tried to change nout of immutable $v to $Δ")

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
function Δnin(t::SizeChangeValidation, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T
    validvisit = !has_visited_in(s, v)
    Δfun = () -> Δnin(base(t), v, Δ..., s=s)
    validate_Δnin(v, Δ, Δfun, validvisit)
end

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


function Δnout(t::SizeChangeValidation, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    validvisit = !has_visited_out(s, v) && !(v in keys(noutΔs(s)))
    Δfun = () -> Δnout(base(t), v, Δ, s=s)
    validate_Δnout(v, Δ, Δfun, validvisit)
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

For more detailed info, see [`ΔSizeGraph`](@ref).
"""
struct ΔSizeInfo
    touch_nin::Dict{AbstractVertex, Vector{Bool}}
    touch_nout::Vector{AbstractVertex}
end
ΔSizeInfo() = ΔSizeInfo(Dict{AbstractVertex, Vector{Bool}}(), AbstractVertex[])

"""
    ΔnoutSizeInfo(v)

Return a `ΔSizeInfo` for the case when nout of `v` is changed, i.e when Δnout(v, Δ) is called.
"""
ΔnoutSizeInfo(v) = Δnout_touches_nin(v, ΔSizeInfo())
"""
    ΔninSizeInfo(v, mask::Bool...)
    ΔninSizeInfo(v, mask::BitArray{1}=falses(length(inputs(v))))

Return a `ΔSizeInfo` for the case when nin of `v` is changed, i.e when Δnin(v, Δ) is called.

Optionally, provide a `mask` where `mask[i] == ismissing(Δ[i])`, i.e `nin(v)[i]` will not be changed.
"""
ΔninSizeInfo(v, mask::Bool) = ΔninSizeInfo(v, BitArray([mask]))
ΔninSizeInfo(v, mask::Bool...) = ΔninSizeInfo(v, BitArray(mask))
ΔninSizeInfo(v, mask::BitArray{1}=falses(length(inputs(v)))) = Δnin_touches_nin(v, ΔSizeInfo(), mask)

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


# "Entry points": Size change originates from here
Δnin_touches_nin(v::AbstractVertex, s, mask::BitArray{1}=falses(length(inputs(v)))) = Δnin_touches_nin(trait(v), v, s, mask)
Δnin_touches_nin(t::DecoratingTrait, v::AbstractVertex, s, mask::BitArray{1}) = Δnin_touches_nin(base(t), v, s, mask)
function Δnin_touches_nin(::SizeAbsorb, v, s, mask::BitArray{1})
    foreach(vi -> Δnout_touches_nin(vi, v, s), inputs(v)[.!mask])
    return s
end
function Δnin_touches_nin(::SizeInvariant, v, s, mask::BitArray{1})
    foreach(vi -> Δnout_touches_nin(vi, v, s), inputs(v)[.!mask])
    foreach(vo -> Δnin_touches_nin(vo, v, s), outputs(v))
    return s
end
function Δnin_touches_nin(::SizeStack, v, s, mask::BitArray{1})
    foreach(vi -> Δnout_touches_nin(vi, v, s), inputs(v)[.!mask])
    foreach(vo -> Δnin_touches_nin(vo, v, s), outputs(v))
    return s
end

# "Propagation points": Size changes here have propagated from somewhere else (from)
function Δnin_touches_nin(v::AbstractVertex, from::AbstractVertex, s)
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


Δnout_touches_nin(v, s) = Δnout_touches_nin(trait(v), v, s)
Δnout_touches_nin(t::DecoratingTrait, v, s) = Δnout_touches_nin(base(t), v, s)
Δnout_touches_nin(::Immutable, v, s) = s
function Δnout_touches_nin(::SizeAbsorb, v, s)
    foreach(vo -> Δnin_touches_nin(vo, v, s), outputs(v))
    return s
end
Δnout_touches_nin(t::SizeTransparent, v::AbstractVertex, s) = Δnin_touches_nin(v, s)


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
"""
    Input

Represents the input direction, i.e coming from the output of another vertex.
"""
struct Input <: Direction end
"""
    Output

Represents the output direction, i.e coming from the input of another vertex.
"""
struct Output <: Direction end
"""
    Both

Represents both directions (`Input` and `Output`).
"""
struct Both <: Direction end

"""
    opposite(d::Direction)

Return the opposite direction of `d`.
"""
opposite(::Input) = Output()
opposite(::Output) = Input()
opposite(b::Both) = b

"""
    ΔSizeGraph

Represents the information on how a size change will propagate as a `MetaDiGraph`.

Each vertex `i` represents a unique `AbstractVertex vi`.
Each edge `e` represents propagation of a size change between vertices `e.src` and `e.dst`.
Edge weights denoted by the symbol `:size` represents the size of the output sent between the vertices.

For the `AbstractVertex vi` associated with vertex `i` in the graph `g`, the following holds `g[i, :vertex] == vi` and `g[vi,:vertex] == i`)

For an edge `e` in graph `g`, the following holds:

If `get_prop(g, e, :direction)` is of type `Output` this means that `Δnout` of `e.dst` is called after processing `e.src`.

If `get_prop(g, e, :direction)` is of type `Input` this means that `Δnin` of `e.dst` is called after processing `e.src`.
"""
function ΔSizeGraph()
    g = MetaDiGraph(0, :size, -1)
    set_indexing_prop!(g, :vertex)
    return g
end

"""
    ΔninSizeGraph(v, mask::Bool...)
    ΔninSizeGraph(v, mask=falses(length(inputs(v))))

Return a `ΔSizeGraph` for the case when nin of `v` is changed, i.e when Δnin(v, Δ) is called.

Optionally, provide a `mask` where `mask[i] == ismissing(Δ[i])`, i.e `nin(v)[i]` will not be changed.
"""
ΔninSizeGraph(v, mask::Bool) = ΔninSizeGraph(v, BitArray([mask]))
ΔninSizeGraph(v, mask::Bool...) = ΔninSizeGraph(v, BitArray(mask))
ΔninSizeGraph(v, mask=falses(length(inputs(v)))) = ΔSizeGraph(Input(), v, mask)
"""
    ΔnoutSizeGraph(v)

Return a `ΔSizeGraph` for the case when nout of `v` is changed, i.e when Δnout(v, Δ) is called.
"""
ΔnoutSizeGraph(v) = ΔSizeGraph(Output(), v)

function ΔSizeGraph(::Input, v, mask=falses(length(inputs(v))))
    g = ΔSizeGraph()
    set_prop!(g, :start, v => Input())
    Δnin_touches_nin(v, g, mask)
end

function ΔSizeGraph(::Output, v)
    g = ΔSizeGraph()
    set_prop!(g, :start, v => Output())
    Δnout_touches_nin(v, g)
end

function LightGraphs.add_edge!(g::MetaDiGraph, src::AbstractVertex, dst::AbstractVertex, d::Direction)
    srcind = vertexind!(g, src)
    dstind = vertexind!(g, dst)
    visited(g, srcind, dstind, d) && return true
    add_edge!(g, srcind, dstind, Dict(:direction => d, g.weightfield => edgesize(d, src, dst)))
    return false
end

function visited(g, srcind, dstind, d)
    has_edge(g, srcind, dstind) && return true
    return visited_out(d, g, dstind)
end

function visited_out(::Output, g, dstind)
    for e in filter_edges(g, :direction, Output())
        e.dst == dstind && return true
    end
    return false
end
visited_out(::Input, g, dstind) = false


edgesize(::Input, src, dst) = nout(src)
edgesize(::Output, src, dst) = nout(dst)

update_state_nout_impl!(g::MetaDiGraph,v,from) = add_edge!(g, from, v, Output())
update_state_nin_impl!(g::MetaDiGraph,v,from)  = add_edge!(g, from, v, Input())

function visited_out(g::MetaDiGraph, v)
    get_prop(g, :start) == (v => Output()) && return true
    ind = vertexind!(g, v)
    return any(e -> e.dst == ind || e.src == ind, filter_edges(g, :direction, Output()))
end

function clear_state_nin!(g::MetaDiGraph, v)
     v in vertexproplist(g, :vertex) || return
     ind = vertexind!(g, v)
     # No need to delete the vertex, only edges matter
     # Furthermore, this function is only called when v anyways shall be in the graph
     for e in filter_edges(g, :direction, Input())
         if e.dst == ind
             rem_edge!(g, e)
         end
     end
end

vertexproplist(g::MetaDiGraph, prop::Symbol) = map(p -> p[:vertex], props.([g], vertices(g)))

function vertexind!(g::MetaDiGraph, v::AbstractVertex,)
    ind = indexin([v], vertexproplist(g, :vertex))[]
    if isnothing(ind)
        add_vertex!(g, :vertex, v)
    end
    return g[v, :vertex]
end

## Generic helper methods end

"""
    SizeDiGraph(g::CompGraph)

Return `cg` as a `MetaDiGraph g`.

Each vertex `i` represents a unique `AbstractVertex vi`.

For the `AbstractVertex vi` associated with vertex `i` in the graph `g`, the following holds `g[i, :vertex] == vi` and `g[vi,:vertex] == i`)

Each edge `e` represents output from `e.src` which is input to `e.dst`.
Edge weights denoted by the symbol `:size` represents the size of the output sent between the vertices.

Note that the order of the input edges to a vertex matters (in case there are more than one) and is not encoded in `g`.
Instead, use `indexin(vi, inputs(v))` to find the index (or indices if input multiple times) of `vi` in `inputs(v)`.
"""
SizeDiGraph(cg::CompGraph) = SizeDiGraph(mapfoldl(v -> flatten(v), (vs1, vs2) -> unique(vcat(vs1, vs2)), cg.outputs))

"""
    SizeDiGraph(v::AbstractVertex)

Return a SizeDiGraph of all parents of v
"""
SizeDiGraph(v::AbstractVertex)= SizeDiGraph(flatten(v))

"""
    SizeDiGraph(vertices::AbstractArray{AbstractVertex,1})

Return a SizeDiGraph of all given vertices
"""
function SizeDiGraph(vertices::AbstractArray{AbstractVertex,1})
    g = MetaDiGraph(0,:size, 0)
    set_indexing_prop!(g, :vertex)
    add_vertices!(g, length(vertices))
    for (ind, v) in enumerate(vertices)
        set_prop!(g, ind, :vertex, v)
        for in_ind in indexin(inputs(v), vertices)
            add_edge!(g, in_ind, ind, :size, nout(vertices[in_ind]))
        end
    end
    return g
end

"""
    fullgraph(v::AbstractVertex)

Return a `SizeDiGraph` of all vertices in the same graph (or connected component) as `v`
"""
fullgraph(v::AbstractVertex) = SizeDiGraph(all_in_graph(v))

"""
    all_in_graph(v::AbstractVertex)

Return an array of vertices in the same graph (or connected component) as `v`
"""
function all_in_graph(v::AbstractVertex, visited = AbstractVertex[])
    v in visited && return visited
    push!(visited, v)
    foreach(vi -> all_in_graph(vi, visited), inputs(v))
    foreach(vo -> all_in_graph(vo, visited), outputs(v))
    return visited
end

"""
    all_in_Δsize_graph(v::AbstractVertex, d::Direction)

Return an array of vertices which will be affected if `v` changes size in direction `d`.
"""
function all_in_Δsize_graph(v::AbstractVertex, d::Direction, visited=AbstractVertex[])
    v in visited && return visited
    push!(visited, v)
    all_in_Δsize_graph(trait(v),d, v, visited)
    return visited
end
function all_in_Δsize_graph(v::AbstractVertex, d::Both, visited=AbstractVertex[])
    all_in_Δsize_graph(v, Input(), visited)
    all_in_Δsize_graph(v, Output(), visited)
    return visited
end


neighbours(::Input, v) = inputs(v)
neighbours(::Output, v) = outputs(v)
neighbours(::Both, v) = vcat(inputs(v), outputs(v))

all_in_Δsize_graph(t::DecoratingTrait, d, v, visited) = all_in_Δsize_graph(base(t), d, v, visited)
function all_in_Δsize_graph(::Immutable, ::Input, v, visited) end
all_in_Δsize_graph(::SizeAbsorb, d, v, visited) = foreach(vn -> all_in_Δsize_graph(vn, opposite(d), visited), neighbours(d, v))
function all_in_Δsize_graph(::SizeTransparent, d, v, visited)
    foreach(vin -> all_in_Δsize_graph(vin, Output(), visited), inputs(v))
    foreach(vout -> all_in_Δsize_graph(vout, Input(), visited), outputs(v))
end

"""
    AbstractJuMPΔSizeStrategy <: AbstractΔSizeStrategy

Abstract type for strategies to change or align the sizes of vertices using JuMP.
"""
abstract type AbstractJuMPΔSizeStrategy <: AbstractΔSizeStrategy end

"""
    ΔSizeFailError <: AbstractJuMPΔSizeStrategy
    ΔSizeFailError(msg::String)

Throws an `ErrorException` with message `msg`.
"""
struct ΔSizeFailError <: AbstractJuMPΔSizeStrategy
    msg::String
end

"""
    ΔSizeFailNoOp <: AbstractJuMPΔSizeStrategy
    ΔSizeFailNoOp()

Does not perform any action.
"""
struct ΔSizeFailNoOp <: AbstractJuMPΔSizeStrategy end

"""
    LogΔSizeExec <: AbstractJuMPΔSizeStrategy
    LogΔSizeExec(msg::String)
    LogΔSizeExec(level::Logging.LogLevel, msg::String)
    LogΔSizeExec(level::Logging.LogLevel, msg::String, andthen::AbstractJuMPΔSizeStrategy)

Throws an `ErrorException` with message `msg`.
"""
struct LogΔSizeExec <: AbstractJuMPΔSizeStrategy
    level::LogLevel
    msg::String
    andthen::AbstractJuMPΔSizeStrategy
end
LogΔSizeExec(msg::String) = LogΔSizeExec(Logging.Info, msg)
LogΔSizeExec(level::LogLevel, msg::String) = LogΔSizeExec(level, msg, ΔSizeFailNoOp())

"""
    DefaultJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy

Default strategy intended to be used when adding some extra constraints or objectives to a model on top of the default.
"""
struct DefaultJuMPΔSizeStrategy <: AbstractJuMPΔSizeStrategy end

struct Exact end
struct Relaxed end

"""
    ΔNout{T} <: AbstractJuMPΔSizeStrategy
    ΔNout{T}(vertex::AbstractVertex, Δ::Integer)
    ΔNoutExact(vertex::AbstractVertex, Δ::Integer, fallback::AbstractJuMPΔSizeStrategy)
    ΔNoutRelaxed(vertex::AbstractVertex, Δ::Integer, fallback::AbstractJuMPΔSizeStrategy)

Strategy for changing nout of `vertex` by `Δ`, i.e new size is `nout(vertex) + Δ`.

If `T == Exact`, size change will be added as a constraint to the model which means that the operation will fail if it is not possible to change `nout(vertex)` by exactly `Δ`. If the operation fails, it will be retried with the `fallback` strategy (default `ΔNoutRelaxed`).

If `T == Relaxed`, size change will be added as an objective to the model which means that `nout(vertex)` might not change by exactly `Δ`. In addition, a constraint that `nout(vertex)` must change is also added.

If the operation fails, it will be retried with the `fallback` strategy (default `ΔNout{Relaxed}` if `T==Exact` and `ΔSizeFailError` if `T==Relaxed`).
"""
struct ΔNout{T} <: AbstractJuMPΔSizeStrategy
    vertex::AbstractVertex
    Δ::Integer
    fallback::AbstractJuMPΔSizeStrategy
end
ΔNoutExact(v::AbstractVertex, Δ::Integer) = ΔNout{Exact}(v, Δ, LogΔSizeExec(Logging.Warn, "Could not change nout of $v by $(Δ)! Relaxing constraints...", ΔNoutRelaxed(v, Δ)))
ΔNoutRelaxed(v::AbstractVertex, Δ::Integer) = ΔNout{Relaxed}(v, Δ, ΔSizeFailError("Could not change nout of $v by $(Δ)!!"))
fallback(s::ΔNout) = s.fallback

"""
    ΔNin{T} <: AbstractJuMPΔSizeStrategy
    ΔNin{T}(vertex::AbstractVertex, Δs::Vector{Maybe{Int}}, fallback::AbstractJuMPΔSizeStrategy)
    ΔNinExact(vertex::AbstractVertex, Δs::Vector{Maybe{Int}})
    ΔNinRelaxed(vertex::AbstractVertex, Δs::Vector{Maybe{Int}})

Strategy for changing nin of `vertex` by `Δs`, i.e new size is `nin(vertex) .+ Δs`. Note that `Δs` must have the same number of elements as `nin(vertex)`.

Use `missing` to indicate "no change required" as 0 will be interpreted as "must not change".

If `T == Exact`, size change will be added as a constraint to the model which means that the operation will fail if it is not possible to change `nin(vertex)` by exactly `Δs`.

If `T == Relaxed`, size change will be added as an objective to the model which means that `nin(vertex)` might not change by exactly `Δs`. In addition, a constraint that `nin(vertex)` must change is also added.

If the operation fails, it will be retried with the `fallback` strategy (default `ΔNin{Relaxed}` if `T==Exact` and `ΔSizeFailError` if `T==Relaxed`).
"""
struct ΔNin{T} <: AbstractJuMPΔSizeStrategy
    vertex::AbstractVertex
    Δs::Vector{Maybe{Int}}
    fallback::AbstractJuMPΔSizeStrategy
    function ΔNin{T}(v, Δs, fallback) where T
        @assert size(Δs) == size(inputs(v)) "Must supply same number of Δs as v has inputs! Got $Δs for $v."
        new(v, Δs, fallback)
    end
end
ΔNinExact(v::AbstractVertex, Δs::Vector{<:Maybe{Int}}) = ΔNin{Exact}(v, Δs, LogΔSizeExec(Logging.Warn, "Could not change nin of $v by $(Δs)! Relaxing constraints...", ΔNinRelaxed(v, Δs)))
ΔNinExact(v::AbstractVertex, Δ::Integer) = ΔNin{Exact}(v, [Δ])
ΔNinRelaxed(v::AbstractVertex, Δs::Vector{<:Maybe{Int}}) = ΔNin{Relaxed}(v, Δs, ΔSizeFailError("Could not change nin of $vertex by $(Δs)!!"))
ΔNinRelaxed(v::AbstractVertex, Δ::Integer) = ΔNin{Relaxed}(v, [Δ])
fallback(s::ΔNin) = s.fallback


"""
    AlignNinToNout <: AbstractJuMPΔSizeStrategy
    AlignNinToNout(vstrat=DefaultJuMPΔSizeStrategy())

Adds variables and constraints for `nin(vi) == nout.(inputs(vi))`.

If it fails, the operation will be retried with the `fallback` strategy (default `ΔSizeFailError`).
"""
struct AlignNinToNout <: AbstractJuMPΔSizeStrategy
    nindict::Dict{AbstractVertex, Vector{JuMP.VariableRef}}
    vstrat::AbstractJuMPΔSizeStrategy
    fallback::AbstractJuMPΔSizeStrategy
end
AlignNinToNout(;vstrat=DefaultJuMPΔSizeStrategy(), fallback=ΔSizeFailError()) = AlignNinToNout(vstrat, fallback)
AlignNinToNout(vstrat, fallback) = AlignNinToNout(Dict{AbstractVertex, JuMP.VariableRef}(), vstrat, fallback)
fallback(s::AlignNinToNout) = s.fallback


"""
    JuMPNorm

Abstract type for norms to a JuMP model.
"""
abstract type JuMPNorm end

"""
    L1NormLinear
    L1NormLinear()

Add a set of linear constraints to a model to map an expression to a variable which is the L1 norm of that expression.
"""
struct L1NormLinear <: JuMPNorm end
"""
    MaxNormLinear
    MaxNormLinear()

Add a set of linear constraints to a model to map an expression to a variable which is the max norm of that expression.
"""
struct MaxNormLinear <: JuMPNorm end

"""
    ScaleNorm{S<:Real,N} <: JuMPNorm
    ScaleNorm(scale, n)

Scales result from `n` with a factor `scale`.
"""
struct ScaleNorm{S<:Real,N<:JuMPNorm} <: JuMPNorm
    scale::S
    n::N
end

"""
    SumNorm{N<:JuMPNorm} <: JuMPNorm

Sum of `ns`.
"""
struct SumNorm{N<:JuMPNorm} <: JuMPNorm
    ns::Vector{N}
end
SumNorm(ns::JuMPNorm...) = SumNorm(collect(ns))
SumNorm(sns::Pair{<:Real, <:JuMPNorm}...) = SumNorm(ScaleNorm.(first.(sns), last.(sns))...)


# Temp methods (hopefully) whose only purpose is to bridge the legacy API
Δnout(::AbstractJuMPΔSizeStrategy, v::AbstractVertex, Δ::Integer; s=nothing) = Δsize(Output(), v, Δ)
Δnin(::AbstractJuMPΔSizeStrategy, v::AbstractVertex, Δs::Maybe{<:Integer}...; s=nothing) = Δsize(Input(), v, Δs...)

"""
    Δsize(d::Direction, v::AbstractVertex, Δ...)

Change size of `v` by `Δ` in direction `d`.
"""
Δsize(d::Input, v, Δs::Maybe{T}...) where T <: Integer = Δsize(ΔNinExact(v, collect(Δs)), all_in_Δsize_graph(v, d))
Δsize(d::Output, v, Δ::T) where T <: Integer = Δsize(ΔNoutExact(v, Δ), all_in_Δsize_graph(v, d))


"""
    Δsize(s::AbstractJuMPΔSizeStrategy, vertices::AbstractArray{<:AbstractVertex})

Calculate new sizes for (potentially) all provided `vertices` using the strategy `s` and apply all changes.
"""
function Δsize(s::AbstractJuMPΔSizeStrategy, vertices::AbstractArray{<:AbstractVertex})
    execute, nins, nouts = newsizes(s, vertices)
    if execute
        Δsize(nins, nouts, vertices)
    end
end

"""
    Δsize(nouts::AbstractVector{<:Integer}, vertices::AbstractVector{<:AbstractVertex})

Set output size of `vertices[i]` to `nouts[i]` for all `i` in `1:length(vertices)`.
Set input size of all keys `vi` in `nins` to `nins[vi]`.
"""
function Δsize(nins::Dict, nouts::AbstractVector{<:Integer}, vertices::AbstractVector{<:AbstractVertex})
    Δnouts = nouts .- nout.(vertices)

    for (i, vi) in enumerate(vertices)
        ninΔs = get(() -> nin(vi), nins, vi) .- nin(vi)
        Δnin_no_prop(vi, ninΔs...)
        Δnout_no_prop(vi, Δnouts[i])
    end

    for (i, vi) in enumerate(vertices)
        ninΔs = get(() -> nin(vi), nins, vi) .- nin(vi)
        after_Δnin(vi, ninΔs...)
        after_Δnout(vi, Δnouts[i])
    end
end

function Δnin_no_prop(v, Δs::Maybe{<:Integer}...)
    any(skipmissing(Δs) .!= 0) || return
    Δnin_no_prop(trait(v), v, Δs)
end
Δnin_no_prop(t::DecoratingTrait, v, Δs) = Δnin_no_prop(base(t), v, Δs)
function Δnin_no_prop(t::SizeChangeLogger, v, Δs)

    @logmsg t.level "Change nin of $(infostr(t, v)) by $(join(compressed_string.(Δs), ", "))"
    Δnin_no_prop(base(t), v, Δs)
end
Δnin_no_prop(::MutationSizeTrait, v, Δs) = Δnin(op(v), Δs...)

function Δnout_no_prop(v, Δ::Integer)
    Δ == 0 && return
    Δnout_no_prop(trait(v), v, Δ)
end
Δnout_no_prop(t::DecoratingTrait, v, Δ) = Δnout_no_prop(base(t), v, Δ)
function Δnout_no_prop(t::SizeChangeLogger, v, Δ)
    @logmsg t.level "Change nout of $(infostr(t, v)) by $(compressed_string(Δ))"
    Δnout_no_prop(base(t), v, Δ)
end
Δnout_no_prop(::MutationSizeTrait, v, Δ) = Δnout(op(v), Δ)

compressed_string(x) = string(x)
struct RangeState
    start
    cnt
end
struct ConsecState
    val
    cnt
end
struct AnyState
    val
end

function form_state(prev, curr)
    Δ = curr - prev
    Δ == 0 && return ConsecState(prev, 2)
    Δ == 1 && return RangeState(prev, 1)
    return AnyState(curr)
end

# Is this.... FP?
increment(h0::RangeState, h1::RangeState, buffer) = RangeState(h0.start, h0.cnt+1)
increment(h0::ConsecState, h1::ConsecState, buffer) = ConsecState(h0.val, h0.cnt+1)
function increment(h0::AnyState, h1::AnyState, buffer)
    write(buffer, "$(h0.val), ")
    return h1
end

function compressed_string(a::AbstractVector)
    length(a) < 20 && return string(a)
    buffer = IOBuffer()
    write(buffer, "[")

    prev = a[1]
    hyp = AnyState(a[1])
    for curr in a[2:end]
        hyp = new_state(hyp, prev, curr, buffer)
        prev = curr
    end
    write_state(hyp, buffer, true)
    write(buffer, "]")
    return String(take!(buffer))
end

new_state(h, prev, curr, buffer) = new_state(h, form_state(prev, curr), buffer)
new_state(h0::T, h1::T, buffer) where T = increment(h0, h1, buffer)
function new_state(h0, h1, buffer)
    write_state(h0, buffer)
    return h1
end

function write_state(h::RangeState, buffer, last=false)
    if h.cnt > 3
        write(buffer, "$(h.start),…, $(h.start + h.cnt)")
    else
        write(buffer, join(string.(h.start:h.start+h.cnt), ", "))
    end
    if !last
        write(buffer, ", ")
    end
end

function write_state(h::ConsecState, buffer, last=false)
    if h.cnt > 3
        write(buffer, "$(h.val)×$(h.cnt)")
    else
        write(buffer, join(repeat([h.val], h.cnt), ", "))
    end
    if !last
        write(buffer, ", ")
    end
end
function write_state(h::AnyState, buffer, last=false)
    if last
        write(buffer, string(h.val))
    end
end

after_Δnin(v, Δs...) = after_Δnin(trait(v), v, Δs)
after_Δnin(t::DecoratingTrait, v, Δs) = after_Δnin(base(t), v, Δs)
after_Δnin(t::SizeChangeValidation, v, Δs) = validate_Δnin(v, Δs, () -> after_Δnin(base(t), v, Δs))
function after_Δnin(t, v, Δs) end

after_Δnout(v, Δ) = after_Δnout(trait(v), v, Δ)
after_Δnout(t::DecoratingTrait, v, Δ) = after_Δnout(base(t), v, Δ)
after_Δnout(t::SizeChangeValidation, v, Δ) = validate_Δnout(v, Δ, () -> after_Δnout(base(t), v, Δ))
function after_Δnout(t, v, Δ) end


newsizes(s::ΔSizeFailError, vertices::AbstractVector{<:AbstractVertex}) = error(s.msg)
newsizes(s::ΔSizeFailNoOp, vertices::AbstractVector{<:AbstractVertex}) = false, Dict(vertices .=> nin.(vertices)), nout.(vertices)
function newsizes(s::LogΔSizeExec, vertices::AbstractVector{<:AbstractVertex})
    @logmsg s.level s.msg
    return newsizes(s.andthen, vertices)
end

"""
    newsizes(s::AbstractJuMPΔSizeStrategy, vertices::AbstractArray{<:AbstractVertex})

Return a vector of new outputs sizes for and a `Dict` of new input sizes for all provided `vertices` using the strategy `s`.

Result vector is index aligned with `vertices`.
Result `Dict` has a vector of input sizes for each element of `vertices` which has an input (i.e everything except input vertices).
"""
function newsizes(s::AbstractJuMPΔSizeStrategy, vertices::AbstractVector{<:AbstractVertex})

    model = sizemodel(s, vertices)

    noutvars = @variable(model, noutvars[i=1:length(vertices)], Int)
    @constraint(model, positive_nonzero_sizes, noutvars .>= 1)

    noutdict = Dict(zip(vertices, noutvars))
    for v in vertices
        vertexconstraints!(v, s, (model=model, noutdict=noutdict))
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
sizemodel(s::AbstractJuMPΔSizeStrategy, vertices) = JuMP.Model(JuMP.with_optimizer(Cbc.Optimizer, loglevel=0))

# Just a short for broadcasting on dicts
getall(d::Dict, ks, deffun=() -> missing) = get.(deffun, [d], ks)

vertexconstraints!(v::AbstractVertex, s, data) = vertexconstraints!(trait(v), v, s, data)
vertexconstraints!(t::DecoratingTrait, v, s, data) = vertexconstraints!(base(t), v, s,data)
function vertexconstraints!(::Immutable, v, s, data)
    @constraint(data.model, data.noutdict[v] == nout(v))
    @constraint(data.model, getall(data.noutdict, inputs(v)) .== nin(v))
end

function vertexconstraints!(v::AbstractVertex, s::AlignNinToNout, data)
    vertexconstraints!(v, s.vstrat, data)
    for vo in outputs(v)
        ninvar = @variable(data.model, integer=true)
        @constraint(data.model, data.noutdict[v] == ninvar)

        ninarr = get!(() -> Vector{JuMP.VariableRef}(undef, length(inputs(vo))), s.nindict, vo)
        ninarr[inputs(vo) .== v] .= ninvar
    end
end

vertexconstraints!(::MutationTrait, v, s, data) = vertexconstraints!(s, v, data)

"""
    vertexconstraints!(s::AbstractJuMPΔSizeStrategy, v, data)

Add constraints for `AbstractVertex v` using strategy `s`.

Extra info like the model and variables is provided in `data`.
"""
function vertexconstraints!(s::AbstractJuMPΔSizeStrategy, v, data)
    ninconstraint!(s, v, data)
    compconstraint!(s, v, (data..., vertex=v))
end

function vertexconstraints!(s::ΔNout{Exact}, v, data)
    # TODO: Replace hardcoded type (DefaultJuMPΔSizeStrategy) with struct field to allow for deeper composition?
    vertexconstraints!(DefaultJuMPΔSizeStrategy(), v, data)
    if v == s.vertex
        # TODO: The name has to go as well so one can compose several ΔNouts
        @constraint(data.model, Δnout_origin, data.noutdict[v] == nout(v) + s.Δ)
    end
end

function vertexconstraints!(s::ΔNin{Exact}, v, data)
    # TODO: Replace hardcoded type (DefaultJuMPΔSizeStrategy) with struct field to allow for deeper composition? Just create one ΔNoutExact for each input for example.
    vertexconstraints!(DefaultJuMPΔSizeStrategy(), v, data)
    if v == s.vertex
        inds = .!ismissing.(s.Δs)
        noutvars = getall(data.noutdict, inputs(v)[inds])
        nins = nin(v)[inds]
        Δs = s.Δs[inds]
        @constraint(data.model, Δnin_origin, noutvars .== nins .+ Δs)
    end
end

"""
    ninconstraint!(s, v, data)

Add input size constraints for `AbstractVertex v` using strategy `s`.

Extra info like the model and variables is provided in `data`.
"""
ninconstraint!(s, v, data) = ninconstraint!(s, trait(v), v, data)
ninconstraint!(s, t::DecoratingTrait, v, data) = ninconstraint!(s, base(t), v, data)
function ninconstraint!(s, ::SizeAbsorb, v, data) end
ninconstraint!(s, ::SizeStack, v, data) = @constraint(data.model, sum(getall(data.noutdict, inputs(v))) == data.noutdict[v])
ninconstraint!(s, ::SizeInvariant, v, data) = @constraint(data.model, getall(data.noutdict, unique(inputs(v))) .== data.noutdict[v])



"""
    ninconstraint!(s, v, data)

Add constraints on the computation (e.g. neural network layer) for `AbstractVertex v` using strategy `s`.

Extra info like the model and variables is provided in `data`.
"""
compconstraint!(s, v::AbstractVertex, data) = compconstraint!(s, base(v), data)
compconstraint!(s, v::CompVertex, data) = compconstraint!(s, v.computation, data)
function compconstraint!(s, f, data) end

"""
    norm!(s::L1NormLinear, model, X)

Add a set of linear constraints to a model to map `X` to an expression `X′` which is the L1 norm of `X`.

Note that it only works for the objective function and only for minimization.
"""
function norm!(s::L1NormLinear, model, X, denom=1)
    # Use trick from http://lpsolve.sourceforge.net/5.1/absolute.htm to make min abs(expression) linear
    X′ = @variable(model, [1:length(X)])
    @constraint(model,  X .<= X′ .* denom)
    @constraint(model, -X .<= X′ .* denom)
    return @expression(model, sum(X′))
end

"""
    norm!(s::L1NormLinear, model, X)

Add a set of linear constraints to a model to map `X` to a variable `X′` which is the max norm of `X`.

Note that it only works for the objective function and only for minimization.
"""
function norm!(s::MaxNormLinear, model, X, denom=1)
    # Use trick from https://math.stackexchange.com/questions/2589887/how-can-the-infinity-norm-minimization-problem-be-rewritten-as-a-linear-program to make min abs(expression) linear
    X′ = @variable(model)
    @constraint(model,  X .<= X′ .* denom)
    @constraint(model, -X .<= X′ .* denom)
    return X′
end

function norm!(s::ScaleNorm, model, X, denom=1)
    X′ = norm!(s.n, model, X, denom)
    return @expression(model, s.scale * X′)
end

norm!(s::SumNorm, model, X, denom=1) = mapfoldl(n -> norm!(n, model, X, denom), (X′,X″) -> @expression(model, X′+X″), s.ns, init=@expression(model, 0))


"""
    sizeobjective!(s::AbstractJuMPΔSizeStrategy, model, noutvars, sizetargets)

Add the objective for `noutvars` using strategy `s`.
"""
sizeobjective!(s::AbstractJuMPΔSizeStrategy, model, noutvars, vertices) = @objective(model, Min, objective!(s, model, noutvars, vertices))

function objective!(s, model, noutvars, vertices)
    sizetargets = nout.(vertices)
    # L1 norm prevents change in vertices which does not need to change.
    # Max norm tries to spread out the change so no single vertex takes most of the change.
    return norm!(SumNorm(0.1 => L1NormLinear(), 0.8 => MaxNormLinear()), model, @expression(model, objective[i=1:length(noutvars)], noutvars[i] - sizetargets[i]), sizetargets)
end

objective!(s::ΔNout{Relaxed}, model, noutvars, vertices) = noutrelax!(model, [s.vertex], [s.Δ], noutvars, vertices)

function objective!(s::ΔNin{Relaxed}, model, noutvars, vertices)
    ininds = .!ismissing.(s.Δs)
    vs = inputs(s.vertex)[ininds]
    return noutrelax!(model, vs, s.Δs[ininds], noutvars, vertices)
end


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

# TODO: Remove since only used for debugging. If only it wasn't so bloody cumbersome to just list the constraints in a JuMP model....
nconstraints(model) = mapreduce(tt -> JuMP.num_constraints.(model,tt...), +,  filter(tt -> tt != (JuMP.VariableRef, MOI.Integer), JuMP.list_of_constraint_types(model)), init=0)

# TODO: Remove since only used for debugging. If only it wasn't so bloody cumbersome to just list the constraints in a JuMP model....
function list_constraints(model)
    for tt in filter(tt -> tt != (JuMP.VariableRef, MOI.Integer), JuMP.list_of_constraint_types(model))
        display(JuMP.all_constraints(model, tt...))
    end
end
