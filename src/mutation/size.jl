

# Vertex traits w.r.t whether size changes propagates
"""
    MutationSizeTrait
Base type for mutation traits relevant to size
"""
abstract type MutationSizeTrait <: MutationTrait end
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



"""
    minΔnoutfactor(v::AbstractVertex)

Returns the smallest n so that allowed changes to nout are n * Z there Z is an integer.
Returns missing if it is not possible to change nout.
"""
minΔnoutfactor(v::AbstractVertex) = minΔnoutfactor(base(v))
minΔnoutfactor(v::InputVertex) = missing
minΔnoutfactor(v::CompVertex) = minΔnoutfactor(v.computation)
minΔnoutfactor(f::Function) = 1 # TODO: Move to test as this does not make alot of sense
minΔnoutfactor(v::MutationVertex) = minΔnoutfactor(trait(v), v)
minΔnoutfactor(t::DecoratingTrait, v::AbstractVertex) = minΔnoutfactor(base(t), v)
minΔnoutfactor(::SizeAbsorb, v::AbstractVertex) = minΔnoutfactor(base(v))
minΔnoutfactor(::SizeInvariant, v::AbstractVertex) = lcm(minΔnoutfactor.(inputs(v)))
function minΔnoutfactor(::SizeStack, v::AbstractVertex)
    absorbing = findterminating(v, inputs)

    # This is not strictly the minimum as using only one of the factors would work as well
    # However, this would create a bias as the same factor would be used all the time
    # Life is really hard sometimes :(

    # Count thingy is for duplicate inputs
    factors = [count(x->x==va,absorbing) * minΔnoutfactor(va) for va in unique(absorbing)]
    isempty(factors) && return 1
    return any(ismissing.(factors)) ? missing : lcm(factors)
end

"""
    minΔninfactor(v::AbstractVertex)

Returns the smallest n so that allowed changes to nin are n * Z there Z is an integer.
Returns missing if it is not possible to change nin.
"""
minΔninfactor(v::AbstractVertex) = minΔninfactor(base(v))
minΔninfactor(v::InputVertex) = missing
minΔninfactor(v::CompVertex) = minΔninfactor(v.computation)
minΔninfactor(f::Function) = 1 # TODO: Move to test as this does not make alot of sense
minΔninfactor(v::MutationVertex) = minΔninfactor(trait(v), v)
minΔninfactor(t::DecoratingTrait, v::AbstractVertex) = minΔninfactor(base(t), v)
minΔninfactor(::SizeAbsorb, v::AbstractVertex) = minΔninfactor(base(v))
minΔninfactor(::SizeInvariant, v::AbstractVertex) = lcm(minΔninfactor.(outputs(v)))
function minΔninfactor(::SizeStack, v::AbstractVertex)
    absorbing = findterminating(v, outputs)
    # This is not strictly the minimum as using only one of the factors would work as well
    # However, this would create a bias as the same factor would be used all the time
    # Life is really hard sometimes :(

    # Count thingy is for duplicate inputs
    factors = [count(x->x==va,absorbing) * minΔninfactor(va) for va in unique(absorbing)]
    isempty(factors) && return 1
    return any(ismissing.(factors)) ? missing : lcm(factors)
end


"""
    findterminating(v::AbstractVertex, f::Function)

Return an array of all vertices which terminate size changes (i.e does not propagate them)
connected through the given function. Will return the given vertex if it is terminating.
"""
findterminating(v::AbstractVertex, f::Function) = findterminating(trait(v), v, f)
findterminating(t::DecoratingTrait, v, f::Function) = findterminating(base(t), v, f)
findterminating(::SizeAbsorb, v, f::Function) = [v]
findterminating(::Immutable, v, f::Function) = [v]
findterminating(::SizeTransparent, v, f::Function) = mapfoldl(vf -> findterminating(vf, f), vcat, f(v), init=[])

"""
    lcmv(arr::AbstractArray{<:Integer})

Remove all elements of arr which are a prime factor of some larger element of arr

# Examples
```julia-repl
julia> import NaiveNASlib:lcmv

julia> lcmv([7,4,6,3,2,1,14])
3-element Array{Int64,1}:
  4
  6
 14
```
"""
function lcmv(arr::AbstractArray{<:Integer})
    srt = sort(arr, rev=true)
    res  = lcmv.(srt[1], srt[2:end]...)
    return arr[findall(x -> x in res, arr)]
end
function lcmv(a::Integer, b::Integer...)
    res = unique(reduce(vcat, lcmv(b...)))
    unique(reduce(vcat, lcmv.(a, res)))
end

function lcmv(a::Integer, b::Integer)
    lm = lcm(a,b)
    return max(a,b) == lm ? lm : [a,b]
end
lcmv() = 1

"""
    VisitState

Memoization struct for traversal when mutating.

Remembers visitation for both forward (in) and backward (out) directions.
"""
struct VisitState{T}
    in::Array{MutationVertex,1}
    out::Array{MutationVertex,1}
    ninΔs::OrderedDict{MutationVertex, Vector{Maybe{T}}}
    noutΔs::OrderedDict{MutationVertex, Maybe{T}}
end
VisitState{T}() where T = VisitState{T}([], [], OrderedDict{MutationVertex, Vector{Maybe{T}}}(), OrderedDict{MutationVertex, Maybe{T}}())
visited_in!(s::VisitState, v::MutationVertex) = push!(s.in, v)
visited_out!(s::VisitState, v::MutationVertex) = push!(s.out, v)
has_visited_in(s::VisitState, v::MutationVertex) = v in s.in
has_visited_out(s::VisitState, v::MutationVertex) = v in s.out

ninΔs(s::VisitState{T}) where T = s.ninΔs
noutΔs(s::VisitState{T}) where T = s.noutΔs

# Only so it is possible to broadcast since broadcasting over dics is reserved
getnoutΔ(defaultfun, s::VisitState{T}, v::MutationVertex) where T = get(defaultfun, s.noutΔs, v)
setnoutΔ!(Δ::T, s::VisitState{T}, v::MutationVertex) where T = s.noutΔs[v] = Δ



Δnin(v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T = Δnin(trait(v), v, Δ..., s=s)
Δnout(v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T = Δnout(trait(v), v, Δ, s=s)

Δnin(t::DecoratingTrait, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T = Δnin(base(t), v, Δ..., s=s)
Δnout(t::DecoratingTrait, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T = Δnout(base(t), v, Δ, s=s)

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

    insizes = deepcopy(nin(v))

    Δnin(op(v), Δ...)
    Δo = concat(insizes, Δ...)
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

function concat(insizes, Δ::Maybe{<:Integer}...)
    return sum(filter(!ismissing, collect(Δ)))
end

function concat(insizes, Δ::Maybe{AbstractArray{T}}...) where T

    res = ismissing(Δ[1]) ? collect(T, 1:insizes[1]) : Δ[1]
    for (innr, Δi) in enumerate(Δ[2:end])
        if ismissing(Δi)
            Δi = collect(1:insizes[innr+1])
        end
        res = vcat(res, map(elem -> elem + sign(elem) * insizes[innr], Δi))
    end

    return res
end

function split_nout_over_inputs(v::AbstractVertex, Δ::T, s::VisitState{T}) where T<:Integer
    # All we want is basically a split of Δ weighted by each individual input size
    # Major annoyance #1: We must comply to the Δfactors so that Δi for input i is an
    # integer multiple of Δfactors[i]
    # Major annoyance #2: Size might already have been propagated to some of the vertices
    # through another path. Need to account for that by digging deeply for non-size-transparent vertices (terminating vertices) and distribute sizes for each one of them to ensure things will work out when we get there. Stuff of nightmares!

    # Note: terminating_vertices is an array of arrays so that terminating_vertices[i] are all terminating vertices seen through input vertex i
    # We will use it later to accumulate all individual size changes in that direction
    inputfilter(v) = v in keys(noutΔs(s)) ? [] : inputs(v)
    terminating_vertices = findterminating.(inputs(v), inputfilter)

    #ftv = flattened_terminating_vertices, okay?
    ftv = vcat(terminating_vertices...)
    uftv = unique(ftv)

    # Find which sizes has not been determined through some other path; those are the ones we shall consider here
    termΔs::AbstractArray{Maybe{T}} = getnoutΔ.(() -> missing, [s], uftv)
    missinginds = ismissing.(termΔs)

    if any(missinginds)
        #Yeah, muftv = missing_unique_flattened_terminating_vertices
        muftv = uftv[missinginds]
        Δ -= sum(skipmissing(termΔs))

        # Remap any duplicated vertices Δf_i => 2 * Δf_i
        Δfactors = Integer[count(x->x==va,ftv) * minΔnoutfactor(va) for va in muftv]
        insizes = nout.(muftv)

        # floor is due to assumption the minimum size is 1 * Δfactors
        limits = floor.(insizes ./ Δfactors)

        objective = function(n)
            # If decreasing (Δ < 0) consider any solution which would result in a nin[i] < Δfactors[i] to be invalid
            Δ < 0 && any(n .>= limits) && return Inf
            # Minimize standard deviation of Δs weighted by input size so larger input sizes get larger Δs
            return std(n .* Δfactors ./ insizes, corrected=false)
        end

        nΔfactors = pick_values(Δfactors, abs(Δ), objective)

        sum(nΔfactors .* Δfactors) == abs(Δ) || @warn "Failed to distribute Δ = $Δ using Δfactors = $(Δfactors) and limits $(limits)!. Proceed with $(nΔfactors) in case receiver can work it out anyways..."

        # Remap any duplicated vertices Δi =>  Δi / 2
        scale_duplicate = Integer[count(x->x==va,ftv) for va in muftv]
        termΔs[missinginds] = div.(sign(Δ) .* nΔfactors .* Δfactors, scale_duplicate)
    end

    # Now its time to accumulate all Δs for each terminating_vertices array. Remember that terminating_vertices[i] is an array of the terminating vertices seen through input vertex i
    vert2size = Dict(uftv .=> termΔs)
    return map(terminating_vertices) do varr
        res = mapreduce(va -> vert2size[va], +, varr)
        return res
    end
end


function split_nout_over_inputs(v::AbstractVertex, Δ::AbstractVector{T}, s::VisitState{<:AbstractVector{T}}) where T<:Integer
    insizes = nin(v)

    Δs = ntuple(i -> T[], length(insizes))

    negind = 1
    function push(elem::Integer, ind::Integer)
        if elem < 0
            push!(Δs[negind], elem)
        elseif elem <= insizes[ind]
            negind = ind
            push!(Δs[ind], elem)
        else
            push(elem - insizes[ind], ind+1)
        end
    end

    foreach(elem -> push(elem, 1), Δ)
    return Δs
end


function Δnin(::SizeInvariant, v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    Δnin(op(v), Δ...)

    Δprop = [Δi for Δi in unique((Δ)) if !ismissing(Δi)]
    @assert length(Δprop) == 1 "Change must be invariant!"

    propagate_nin(v, Δprop...; s=s)
    propagate_nout(v, repeat(Δprop, length(inputs(v)))...; s=s)
end

function Δnout(::SizeInvariant, v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    Δnout(op(v), Δ)

    propagate_nin(v, Δ, s=s)
    propagate_nout(v, fill(Δ, length(inputs(v)))...; s=s)
end

## Generic helper methods

function invisit(v::MutationVertex, s::VisitState{T}) where T
    has_visited_in(s, v) && return true
    visited_in!(s, v)
    return false
end

function outvisit(v::MutationVertex, s::VisitState{T}) where T
    has_visited_out(s, v) && return true
    visited_out!(s, v)
    return false
end

function anyvisit(v::MutationVertex, s::VisitState{T}) where T
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
    # If not, the "if first" block in the end will propagate anyways, leaving
    # it up to each vertex implementation to handle the missing value.
    # See testset "Transparent residual fork block" for a motivation
    first = isempty(ninΔs(s))
    for vi in outputs(v)
        ins = inputs(vi)
        Δs = get!(ninΔs(s), vi) do
            Array{Union{Missing, T},1}(missing, length(ins))
        end
        # Add Δ for each input which is the current vertex (at least and typically one)
        foreach(ind -> Δs[ind] = Δ, findall(vx -> vx == v, ins))
        any(ismissing.(Δs)) || Δnin(vi, Δs...; s=s)
    end

    if first
        # Must delete from contexts before further traversal and we don't
        # wanna delete from the collection we're iterating over
        # New contexts might be added as we traverse though, so after
        # we are done with the current batch we need to check again
        # if there are new contexts, hence the while loop
        while !isempty(ninΔs(s))
            tmpctxs = copy(ninΔs(s))
            for (vn, ctx) in tmpctxs
                delete!(ninΔs(s), vn)
                # Note: Other contexts may be "completed" as a consequence of this
                # However, if that happens the call below should just return immediately
                # due to vertex having been visited.
                # Should be safe to just add a check for this here or maybe remove
                # completed contexts in the above loop
                Δnin(vn, ctx..., s=s)
            end
        end
    end
end

function propagate_nout(v::MutationVertex, Δ::T...; s::VisitState{T}=VisitState{T}()) where T
    setnoutΔ!.(Δ, [s], inputs(v))
    for (Δi, vi) in zip(Δ, inputs(v))
        Δnout(vi, Δi; s=s)
    end
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


## Generic helper methods end
