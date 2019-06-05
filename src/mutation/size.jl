
"""
    minΔinfactor(v::AbstractVertex)

Returns the smallest n so that allowed changes to nin are n * Z there Z is an integer.
Returns missing if it is not possible to change nin.
"""
minΔinfactor(v::AbstractVertex) = minΔinfactor(base(v))
minΔinfactor(v::InputVertex) = missing
minΔinfactor(v::CompVertex) = minΔinfactor(v.computation)
minΔinfactor(f::Function) = 1 # TODO: Move to test as this does not make alot of sense
minΔinfactor(v::InvariantVertex) = maximum(minΔinfactor.(inputs(v)))
function minΔinfactor(v::StackingVertex)
    absorbing = findabsorbing(v, inputs)
    return maximum([count(x->x==va,absorbing) * minΔinfactor(va) for va in unique(absorbing)])
end

# Vertex traits w.r.t whether size changes propagates
struct Transparent end
struct Absorb end
sizetransparency(v::AbstractVertex) = Absorb()
sizetransparency(v::Union{InvariantVertex, StackingVertex}) = Transparent()

"""
    findabsorbing(v::AbstractVertex, f::Function)

Return an array of all vertices which absorb size changes (i.e does not require nin==nout)
connected through the given function. Will return the given vertex if it is absorbing
"""
findabsorbing(v::AbstractVertex, f::Function) = findabsorbing(sizetransparency(v), v, f)
findabsorbing(::Absorb, v, f::Function) = [v]
findabsorbing(::Transparent, v, f::Function) = mapfoldl(vf -> findabsorbing(vf, f), vcat, f(v), init=[])


nin(v::AbsorbVertex) = nin(v.state)
nout(v::AbsorbVertex) = nout(v.state)
nout(v::StackingVertex) = nout(v.state)
nin(v::StackingVertex) = nin(v.state)
nout(v::InvariantVertex) = nin(v)[1]
nin(v::InvariantVertex) = nout.(inputs(v))
nin(v::InputSizeVertex) = v.size
nout(v::InputSizeVertex) = v.size


"""
    VisitState

Memoization struct for traversal when mutating.

Remembers visitation for both forward (in) and backward (out) directions.
"""
struct VisitState{T}
    in::Array{AbstractMutationVertex,1}
    out::Array{AbstractMutationVertex,1}
    contexts::OrderedDict{AbstractMutationVertex, Vector{Maybe{T}}}
end
VisitState{T}() where T = VisitState{T}([], [], OrderedDict{AbstractMutationVertex, Vector{Maybe{T}}}())
visited_in!(s::VisitState, v::AbstractMutationVertex) = push!(s.in, v)
visited_out!(s::VisitState, v::AbstractMutationVertex) = push!(s.out, v)
has_visited_in(s::VisitState, v::AbstractMutationVertex) = v in s.in
has_visited_out(s::VisitState, v::AbstractMutationVertex) = v in s.out

no_context(s::VisitState{T}) where T = isempty(s.contexts)
context!(defaultfun, s::VisitState{T}, v::AbstractMutationVertex) where T = get!(defaultfun, s.contexts, v)
delete_context!(s::VisitState{T}, v::AbstractMutationVertex) where T = delete!(s.contexts, v)
contexts(s::VisitState{T}) where T = s.contexts

function Δnin(v::AbsorbVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T
    invisit(v, s) && return
    Δnin(v.state, Δ...)
    propagate_nout(v, Δ..., s=s)
end

function Δnout(v::AbsorbVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    outvisit(v,s) && return
    Δnout(v.state, Δ)
    propagate_nin(v, Δ, s=s)
end


function Δnin(v::StackingVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    insizes = deepcopy(nin(v))

    Δnin(v.state, Δ...)
    Δo = concat(insizes, Δ...)
    Δnout(v.state, Δo)

    propagate_nin(v, Δo; s=s)
end

function Δnout(v::StackingVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    Δnout(v.state, Δ)

    propagate_nin(v, Δ, s=s) # If there are multiple outputs they must all be updated
    Δs = split_nout_over_inputs(v, Δ)
    Δnin(v.state, Δs...)
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

function split_nout_over_inputs(v::StackingVertex, Δ::T) where T<:Integer
    insizes = nin(v)

    # We basically want a split of Δ weighted by each individual input size:
    Δs = round.(typeof(Δ), insizes .* Δ / sum(insizes))

    # However, we can't set any of the input sizes to 0
    # TODO: Add min_nout and min_nin functions to ensure the above is always
    # possible
    Δs = max.(Δs, -insizes .+ 1) #1 is the minimum input size
    Δdeficit = sum(Δs) - Δ
    for _ in 1:Δdeficit
        Δs[argmax(insizes .+ Δs)] -= 1
    end
    return Δs
end

function split_nout_over_inputs(v::StackingVertex, Δ::AbstractArray{T,1}) where T<:Integer
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


function Δnin(v::InvariantVertex, Δ::Maybe{T}...; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    Δnin(v.op, Δ...)

    Δprop = [Δi for Δi in unique((Δ)) if !ismissing(Δi)]
    @assert length(Δprop) == 1 "Change must be invariant!"

    propagate_nin(v, Δprop...; s=s)
    propagate_nout(v, repeat(Δprop, length(inputs(v)))...; s=s)
end

function Δnout(v::InvariantVertex, Δ::T; s::VisitState{T}=VisitState{T}()) where T
    anyvisit(v, s) && return

    Δnout(v.op, Δ)

    propagate_nin(v, Δ, s=s)
    propagate_nout(v, fill(Δ, length(inputs(v)))...; s=s)
end

## Generic helper methods

function invisit(v::AbstractMutationVertex, s::VisitState{T}) where T
    has_visited_in(s, v) && return true
    visited_in!(s, v)
    return false
end

function outvisit(v::AbstractMutationVertex, s::VisitState{T}) where T
    has_visited_out(s, v) && return true
    visited_out!(s, v)
    return false
end

function anyvisit(v::AbstractMutationVertex, s::VisitState{T}) where T
    in = invisit(v, s)
    out = outvisit(v, s)
    return in || out
end

function propagate_nin(v::AbstractMutationVertex, Δ::T; s::VisitState{T}) where T
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

function propagate_nout(v::AbstractMutationVertex, Δ::T...; s::VisitState{T}=VisitState{T}()) where T
    for (Δi, vi) in zip(Δ, inputs(v))
        Δnout(vi, Δi; s=s)
    end
end

## Generic helper methods end
