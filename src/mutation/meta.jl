"""
    MutationOp
Perform mutation operations
"""
abstract type MutationOp end
abstract type MutationState <: MutationOp end

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
Δnin(s::NoOp, Δ...) = Δ
Δnout(s::NoOp, Δ) = Δ

"""
    InvSize
Invariant size, i.e nin == nout
"""
mutable struct InvSize <: MutationState
    size::Integer
end
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
IoSize(in::Integer, out::Integer) = IoSize([in], out)
nin(s::IoSize) = s.nin
nout(s::IoSize) = s.nout
Δnin(s::IoSize, Δ::Integer...) = s.nin .+= Δ
Δnout(s::IoSize, Δ::Integer) = s.nout += Δ


"""
    IoIndices
Indexes to retain for input and output of a computation

What those indices mean is up to the computation
"""
mutable struct IoIndices <: MutationState
    in::AbstractArray{<:AbstractArray{<:Integer,1},1}
    out::AbstractArray{<:Integer,1}
end
IoIndices(in::Integer, out::Integer) = IoIndices([collect(1:in)], collect(1:out))
nin(s::IoIndices) = length(s.in)
nout(s::IoIndices) = length(s.out)
Δnin(s::IoIndices, Δ::AbstractArray{<:Integer,1}...) = s.in = collect(Δ)
Δnout(s::IoIndices, Δ::AbstractArray{<:Integer,1}) = s.out = Δ
