"""
    VertexMeta
Metadata about vertex needed to perform mutation operations
"""
abstract type VertexMeta end

"""
    nin(c::VertexMeta)

Returns the size of the input data to the vertex

Computation may fail if input does not have size nin.
"""
function nin end

"""
    nout(c::VertexMeta)

Returns the size of the output data from the vertex.
"""
function nout end


"""
    Δnin(v::VertexMeta, Δ)
    Δnin(v::AbstractMutationVertex, Δ)

Change input size by Δ. New size is nin + Δ

Might propagate to other vertices when performed on an AbstractMutationVertex
depending on vertex type
"""
function Δnin end

"""
    Δnout(v::VertexMeta, Δ)
    Δnout(v::AbstractMutationVertex, Δ)

Change output size by Δ. New size is nout + Δ

Might propagate to other vertices when performed on an AbstractMutationVertex
depending on vertex type
"""
function Δnout end

"""
    InvSize
Invariant size, i.e nin == nout
"""
mutable struct InvSize <: VertexMeta
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
mutable struct IoSize <: VertexMeta
    nin::AbstractArray{Integer,1}
    nout::Integer
end
IoSize(in::Integer, out::Integer) = IoSize([in], out)
nin(s::IoSize) = s.nin
nout(s::IoSize) = s.nout
Δnin(s::IoSize, Δ::Integer...) = s.nin .+= Δ
Δnout(s::IoSize, Δ::Integer) = s.nout += Δ
