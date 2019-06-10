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

Returns the size of the input data to the vertex

Computation may fail if input does not have size nin.
"""
function nin end

"""
    nout(m::MutationState)
    nout(v::AbstractMutationVertex)

Returns the size of the output data from the vertex.
"""
function nout end


"""
    Δnin(v::MutationOp, Δ)
    Δnin(v::AbstractMutationVertex, Δ)

Change input size by Δ. New size is nin + Δ

Might propagate to other vertices when performed on an AbstractMutationVertex
depending on vertex type
"""
function Δnin end

"""
    Δnout(v::MutationOp, Δ)
    Δnout(v::AbstractMutationVertex, Δ)

Change output size by Δ. New size is nout + Δ

Might propagate to other vertices when performed on an AbstractMutationVertex
depending on vertex type
"""
function Δnout end

"""
    NoOp
Does not perform any operations
"""
struct NoOp <: MutationOp end
clone(op::NoOp) = op
Δnin(s::NoOp, Δ...) = Δ
Δnout(s::NoOp, Δ) = Δ

"""
    InvSize
Invariant size, i.e nin == nout
"""
mutable struct InvSize <: MutationState
    size::Integer
end
clone(s::InvSize) = InvSize(s.size)

nin(s::InvSize) = s.size
nout(s::InvSize) = s.size
function Δnin(s::InvSize, Δ::Integer...)
    @assert length(Δ) == 1 "Must be single input! Got $Δ"
    Δnout(s, Δ[1])
end
Δnout(s::InvSize, Δ::Integer) = s.size += Δ
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
    InvIndices
Invariant size, i.e nin == nout.
"""
mutable struct InvIndices <: MutationState
    inds::AbstractArray{<:Integer,1}
end
InvIndices(size::Integer) = InvIndices(1:size)
InvIndices(s::InvSize) = InvIndices(nin(s))
clone(s::InvIndices) = InvIndices(copy(s.inds))

nin(s::InvIndices) = [nout(s)]
nout(s::InvIndices) = length(s.inds)

function Δnin(s::InvIndices, Δ::AbstractArray{<:Integer,1}...)
    @assert length(Δ) == 1 "Must be single input! Got $Δ"
    Δnout(s, Δ[1])
end
Δnout(s::InvIndices, Δ::AbstractArray{<:Integer,1}) = s.inds = copy(Δ)

function reset!(s::InvIndices)
    s.inds[1:end] = 1:length(s.inds)
end


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
