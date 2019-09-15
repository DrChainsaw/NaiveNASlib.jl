"""
    MutationOp
Perform mutation operations
"""
abstract type MutationOp end
abstract type MutationState <: MutationOp end

function reset_in!(s::MutationOp) end
function reset_out!(s::MutationOp) end
function reset!(s::MutationOp) end

"""
    nin(m::MutationState)
    nin(v::AbstractMutationVertex)

Return the size of the input data to the vertex

Computation may fail if input does not have size nin.
"""
function nin end

"""
    nout(m::MutationState)
    nout(v::AbstractMutationVertex)

Return the size of the output data from the vertex.
"""
function nout end

"""
    nin_org(m::MutationState)
    nin_org(v::AbstractMutationVertex)

Return the size of the input data to the vertex before any mutation was performed.

Computation may fail if input does not have size nin.
Computation may also fail if vertex has MutationState which does not track changes.
"""
function nin_org end

"""
    nout_org(m::MutationState)
    nout_org(v::AbstractMutationVertex)

Return the size of the output data from the vertex before any mutation was performed.
Computation may fail if vertex has MutationState which does not track changes.
"""
function nout_org end


"""
    Δnin(o::MutationOp, Δ)
    Δnin(v::AbstractMutationVertex, Δ)

Change input size by Δ. New size is nin + Δ

Might propagate to other vertices when performed on an AbstractMutationVertex
depending on vertex type
"""
function Δnin end

"""
    Δnout(o::MutationOp, Δ)
    Δnout(v::AbstractMutationVertex, Δ)

Change output size by Δ. New size is nout + Δ

Might propagate to other vertices when performed on an AbstractMutationVertex
depending on vertex type
"""
function Δnout end

"""
    in_inds(m::MutationState)

Return selected input indices.
"""
function in_inds end

"""
    out_inds(m::MutationState)

Return selected output indices.
"""
function out_inds end

"""
    NoOp
Does not perform any operations
"""
struct NoOp <: MutationOp end
clone(op::NoOp) = op
Δnin(s::NoOp, Δ...) = Δ
Δnout(s::NoOp, Δ) = Δ

"""
    IoSize
Size for input and output of a computation
"""
mutable struct IoSize <: MutationState
    nin::AbstractArray{Integer,1}
    nout::Integer
end
IoSize(size::Integer) = IoSize(size, size)
IoSize(in::Integer, out::Integer) = IoSize([in], out)
clone(s::IoSize) = IoSize(copy(nin(s)), nout(s))

nin(s::IoSize) = s.nin
nout(s::IoSize) = s.nout

in_inds(s::IoSize) = [1:insize for insize in s.nin]
out_inds(s::IoSize) = 1:s.nout

Maybe{T} = Union{T, Missing}
Δnin(s::IoSize, Δ::Maybe{Integer}...) = s.nin .+= replace(collect(Δ), missing => 0)
Δnout(s::IoSize, Δ::Integer) = s.nout += Δ
function Δnin(s::IoSize, Δ::Maybe{AbstractArray{<:Integer,1}}...)
    for (i,Δi) in enumerate(Δ)
        if !ismissing(Δi)
            s.nin[i] = length(Δi)
        end
    end
end
Δnout(s::IoSize, Δ::AbstractArray{<:Integer,1}) = s.nout = length(Δ)

"""
    IoIndices
Indexes to retain for input and output of a computation

What those indices mean is up to the computation
"""
mutable struct IoIndices <: MutationState
    in::AbstractArray{<:AbstractArray{<:Integer,1},1}
    out::AbstractArray{<:Integer,1}
end
IoIndices(in::Integer, out::Integer) = IoIndices([in], out)
IoIndices(in::AbstractArray{<:Integer}, out::Integer) = IoIndices(collect.(range.(1, in, step=1)), collect(1:out))
IoIndices(s::MutationState) = IoIndices(nin(s), nout(s))
clone(s::IoIndices) = IoIndices(deepcopy(s.in), copy(s.out))

nin(s::IoIndices) = length.(s.in)
nout(s::IoIndices) = length(s.out)

in_inds(s::IoIndices) = s.in
out_inds(s::IoIndices) = s.out

Δnin(s::IoIndices, Δ::Maybe{AbstractArray{<:Integer,1}}...) = Δnin(s::IoIndices, map(i -> ismissing(Δ[i]) ? s.in[i] : Δ[i], eachindex(Δ))...)
Δnin(s::IoIndices, Δ::AbstractArray{<:Integer,1}...) = s.in = collect(deepcopy(Δ))
Δnout(s::IoIndices, Δ::AbstractArray{<:Integer,1}) = s.out = copy(Δ)
function reset_in!(s::IoIndices)
    foreach(s.in) do ini
        ini[1:end] = 1:length(ini)
    end
end
function reset_out!(s::IoIndices)
    s.out[1:end] = 1:length(s.out)
end

"""
    IoChange

Size for input and output of a computation stored as a change towards an original size.

Also maintains indices separately.

Useful for tracking change through several mutations before making a selection of which parameters to keep.
"""
mutable struct IoChange <: MutationState
    size::MutationState
    indices::MutationState
    inΔ::AbstractVector{<:Integer}
    outΔ::Integer
end

IoChange(insize, outsize) =  IoChange(IoSize(insize, outsize), IoIndices(insize, outsize))
function IoChange(size::MutationState, inds::MutationState)
    nin(size) == nin(inds) || error("Insizes differ: $(nin(size)) vs $(nin(inds))")
    nout(size) == nout(inds) || error("Outsizes differ: $(nout(size)) vs $(nout(inds))")
    IoChange(size, inds, zeros(eltype(nin(size)), length(nin(size))), 0)
end
clone(s::IoChange) = IoChange(clone(s.size), clone(s.indices), copy(s.inΔ), s.outΔ)

nin(s::IoChange) = nin_org(s) .+ s.inΔ
nout(s::IoChange) = nout(s.size) + s.outΔ

in_inds(s::IoChange) = trunc_or_pad.(in_inds(s.indices), nin(s))
out_inds(s::IoChange) = trunc_or_pad(out_inds(s.indices), nout(s))

nin_org(s::IoChange) = nin(s.size)
nout_org(s::IoChange) = nout(s.size)

function trunc_or_pad(vec, size)
    res = -ones(eltype(vec), size)
    inds = 1:min(size, length(vec))
    res[inds] = vec[inds]
    return res
end

Δnin(s::IoChange, Δ::Maybe{Integer}...) = s.inΔ .+= replace(collect(Δ), missing => 0)
Δnout(s::IoChange, Δ::Integer) = s.outΔ += Δ
function Δnin(s::IoChange, Δ::Maybe{AbstractArray{<:Integer,1}}...)
    Δnin(s.indices, Δ...)

    for (i, Δi) in enumerate(Δ)
        if !ismissing(Δi)
            s.inΔ[i] = length(Δi) - nin(s.size)[i]
        end
    end
end

function Δnout(s::IoChange, Δ::AbstractArray{<:Integer,1})
    Δnout(s.indices, Δ)
    s.outΔ = nout(s.indices) - nout(s.size)
end

function reset_in!(s::IoChange)
    Δnin(s.size, s.inΔ...)
    s.inΔ[1:end] .= 0
    Δnin(s.indices, collect.(StepRange.(1,1, nin(s)))...)
    reset_in!(s.size)
    reset_in!(s.indices)
end

function reset_out!(s::IoChange)
    Δnout(s.size, s.outΔ)
    s.outΔ = 0
    Δnout(s.indices, collect(1:nout(s)))
    reset_out!(s.size)
    reset_out!(s.indices)
end
