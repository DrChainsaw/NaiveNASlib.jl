
"""
AbstractMutationVertex

Base type for vertices which can be mutated
"""
abstract type AbstractMutationVertex <: AbstractVertex end

"""
    base(v::AbstractMutationVertex)

Return base vertex
"""
function base end

inputs(v::AbstractMutationVertex)  = inputs(base(v))
outputs(v::AbstractMutationVertex) = outputs(base(v))

Maybe{T} = Union{T, Missing}
"""
    VisitState

Memoization struct for traversal when mutating.

Remembers visitation for both forward (in) and backward (out) directions.
"""
struct VisitState{T}
    in::Array{AbstractMutationVertex,1}
    out::Array{AbstractMutationVertex,1}
    contexts::Dict{AbstractMutationVertex, Vector{Maybe{T}}}
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
    # If not, the "if first" block in the end will insert zeroes for the missing
    # inputs and propagate anyways.
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
        for (v, ctx) in contexts(s)
            delete_context!(s, v)
            # Note: Other contexts may be "completed" as a consequence of this
            # However, if that happens the call below should just return immediately
            # due to vertex having been visited.
            # Should be safe to just add a check for this here or maybe remove
            # completed contexts in the above loop
            Δnin(v, ctx..., s=s)
        end
    end
end

function propagate_nout(v::AbstractMutationVertex, Δ::T...; s::VisitState{T}=VisitState{T}()) where T
    for (Δi, vi) in zip(Δ, inputs(v))
        Δnout(vi, Δi; s=s)
    end
end

## Generic helper methods end

"""
    OutputsVertex

Decorates an AbstractVertex with output edges.
"""
struct OutputsVertex <: AbstractVertex
    base::AbstractVertex
    outs::AbstractArray{AbstractVertex,1}
end
OutputsVertex(v::AbstractVertex) = OutputsVertex(v, AbstractVertex[])
init!(v::OutputsVertex, p::AbstractVertex) = foreach(in -> push!(outputs(in), p), inputs(v))
base(v::OutputsVertex) = v.base
(v::OutputsVertex)(x...) = base(v)(x...)

inputs(v::OutputsVertex) = inputs(base(v))
outputs(v::OutputsVertex) = v.outs

"""
    InputSizeVertex

Vertex with an (immutable) size. Intended use if for wrapping an InputVertex
in conjuntion with mutation
"""
struct InputSizeVertex <: AbstractVertex
    base::AbstractVertex
    size::Integer

    function InputSizeVertex(b::OutputsVertex, size::Integer)
        this = new(b, size)
        init!(b, this)
        return this
    end
end
InputSizeVertex(b::AbstractVertex, size::Integer) = InputSizeVertex(OutputsVertex(b), size)
base(v::InputSizeVertex)::AbstractVertex = v.base
(v::InputSizeVertex)(x...) = base(v)(x...)

inputs(v::InputSizeVertex) = inputs(base(v))
outputs(v::InputSizeVertex) = outputs(base(v))

nin(v::InputSizeVertex) = v.size
nout(v::InputSizeVertex) = v.size


"""
    AbsorbVertex

Vertex which absorbs changes in nout or nin. An example of this is a vertex
which multiplies its input with an nin x nout matrix.
"""
struct AbsorbVertex <: AbstractMutationVertex
    base::AbstractVertex
    state::MutationState

    function AbsorbVertex(b::OutputsVertex, state::MutationState)
        this = new(b, state)
        init!(b, this)
        return this
    end
end
AbsorbVertex(b::AbstractVertex, state::MutationState) = AbsorbVertex(OutputsVertex(b), state)
base(v::AbsorbVertex)::AbstractVertex = v.base

nin(v::AbsorbVertex) = nin(v.state)
nout(v::AbsorbVertex) = nout(v.state)

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


"""
    StackingVertex

Vertex which is transparent w.r.t mutation.

Size of output is sum of sizes of inputs. Examples of computations are scalar operations
(e.g add x to every element) and concatenation.
"""
struct StackingVertex <: AbstractMutationVertex
    base::AbstractVertex
    state::MutationState

    function StackingVertex(b::OutputsVertex, op::MutationState)
        this = new(b, op)
        init!(b, this)
        return this
    end
end
StackingVertex(b::AbstractVertex) = StackingVertex(OutputsVertex(b))
StackingVertex(b::Union{OutputsVertex, AbstractMutationVertex}) = StackingVertex(b, IoSize(nout.(inputs(b)), sum(nout.(inputs(b)))))
base(v::StackingVertex)::AbstractVertex = v.base

nout(v::StackingVertex) = nout(v.state)
nin(v::StackingVertex) = nin(v.state)

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

"""
    InvariantVertex

Vertex which is size invariant in the sense that all inputs and outputs have the same size.

Examples of computations are scalar and element wise operations.
"""
struct InvariantVertex <: AbstractMutationVertex
    base::AbstractVertex
    op::MutationOp

    function InvariantVertex(b::OutputsVertex, op::MutationOp)
        this = new(b, op)
        init!(b, this)
        return this
    end
end
InvariantVertex(b::AbstractVertex) = InvariantVertex(OutputsVertex(b), NoOp())
base(v::InvariantVertex)::AbstractVertex = v.base

nout(v::InvariantVertex) = nin(v)[1]
nin(v::InvariantVertex) = nout.(inputs(v))

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
