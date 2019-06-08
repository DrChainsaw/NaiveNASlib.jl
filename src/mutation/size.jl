

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

nin(v::InputSizeVertex) = v.size
nin(v::MutationVertex) = nin(trait(v), v, op(v))
nin(::MutationTrait, v::AbstractVertex, op::MutationState) = nin(op)
# SizeInvariant does not need mutation state to keep track of sizes
nin(::SizeInvariant, v::AbstractVertex, op::MutationOp) = nout.(inputs(v))

nout(v::InputSizeVertex) = v.size
nout(v::MutationVertex) = nout(trait(v), v, op(v))
nout(::MutationTrait, v::AbstractVertex, op::MutationState) = nout(op)
# SizeInvariant does not need mutation state to keep track of sizes
nout(t::SizeInvariant, v::AbstractVertex, op::MutationOp) = nin(v)[1]



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
minΔnoutfactor(::SizeAbsorb, v::AbstractVertex) = minΔnoutfactor(base(v))
minΔnoutfactor(::SizeInvariant, v::AbstractVertex) = maximum(minΔnoutfactor.(inputs(v)))
function minΔnoutfactor(::SizeStack, v::AbstractVertex)
    absorbing = findabsorbing(v, inputs)
    return maximum([count(x->x==va,absorbing) * minΔnoutfactor(va) for va in unique(absorbing)])
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
minΔninfactor(::SizeAbsorb, v::AbstractVertex) = minΔninfactor(base(v))
minΔninfactor(::SizeInvariant, v::AbstractVertex) = maximum(minΔninfactor.(outputs(v)))
function minΔninfactor(::SizeStack, v::AbstractVertex)
    absorbing = findabsorbing(v, outputs)
    return maximum([count(x->x==va,absorbing) * minΔninfactor(va) for va in unique(absorbing)])
end


"""
    findabsorbing(v::AbstractVertex, f::Function)

Return an array of all vertices which absorb size changes (i.e does not require nin==nout)
connected through the given function. Will return the given vertex if it is absorbing
"""
findabsorbing(v::AbstractVertex, f::Function) = findabsorbing(trait(v), v, f)
findabsorbing(::SizeAbsorb, v, f::Function) = [v]
findabsorbing(::Immutable, v, f::Function) = [v] #TODO!!!
findabsorbing(::SizeTransparent, v, f::Function) = mapfoldl(vf -> findabsorbing(vf, f), vcat, f(v), init=[])


"""
    VisitState

Memoization struct for traversal when mutating.

Remembers visitation for both forward (in) and backward (out) directions.
"""
struct VisitState{T}
    in::Array{MutationVertex,1}
    out::Array{MutationVertex,1}
    contexts::OrderedDict{MutationVertex, Vector{Maybe{T}}}
end
VisitState{T}() where T = VisitState{T}([], [], OrderedDict{MutationVertex, Vector{Maybe{T}}}())
visited_in!(s::VisitState, v::MutationVertex) = push!(s.in, v)
visited_out!(s::VisitState, v::MutationVertex) = push!(s.out, v)
has_visited_in(s::VisitState, v::MutationVertex) = v in s.in
has_visited_out(s::VisitState, v::MutationVertex) = v in s.out

no_context(s::VisitState{T}) where T = isempty(s.contexts)
context!(defaultfun, s::VisitState{T}, v::MutationVertex) where T = get!(defaultfun, s.contexts, v)
delete_context!(s::VisitState{T}, v::MutationVertex) where T = delete!(s.contexts, v)
contexts(s::VisitState{T}) where T = s.contexts


Δnin(v::AbstractVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T = Δnin(trait(v), v, Δ..., s=s)
Δnout(v::AbstractVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T = Δnout(trait(v), v, Δ, s=s)

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
    Δs = split_nout_over_inputs(v, Δ)
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

function split_nout_over_inputs(v::AbstractVertex, Δ::T) where T<:Integer
    insizes = nin(v)
    Δfactors = minΔnoutfactor.(inputs(v))

    # All we want is basically a split of Δ weighted by each individual input size
    # Major annoyance: We must comply to the Δfactors so that Δi for input i is an
    # integer multiple of Δfactors[i]
    limits = ceil.(insizes ./ Δfactors)

    objective = function(n)
        Δ < 0 && any(n .>= limits) && return Inf
        return std(n .* Δfactors ./ insizes, corrected=false)
    end

    nΔfactors = pick_values(Δfactors, abs(Δ), objective)

    sum(nΔfactors .* Δfactors) == abs(Δ) || @warn "Failed to distribute Δ = $Δ using Δfactors = $(Δfactors)!. Proceed with $(nΔfactors) in case receiver can work it out anyways..."

    Δs = sign(Δ) .* nΔfactors .* Δfactors
end


function split_nout_over_inputs(v::AbstractVertex, Δ::AbstractArray{T,1}) where T<:Integer
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
    first = no_context(s)
    for vi in outputs(v)
        ins = inputs(vi)
        Δs = context!(s, vi) do
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
        while !no_context(s)
            tmpctxs = copy(contexts(s))
            for (vn, ctx) in tmpctxs
                delete_context!(s, vn)
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
    for (Δi, vi) in zip(Δ, inputs(v))
        Δnout(vi, Δi; s=s)
    end
end


# Yeah, this is an exhaustive search. Non-linear objective (such as std) makes
# DP not feasible (or maybe I'm just too stupid to figure out the optimal substructure).
# Shouldn't matter as any remotely sane application of this library should call this function
# with relatively small targets and few values
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
