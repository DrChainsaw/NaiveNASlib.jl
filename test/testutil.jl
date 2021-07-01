
function implementations(T::Type)
    return mapreduce(t -> isabstracttype(t) ? implementations(t) : t, vcat, subtypes(T), init=[])
end

issame(v1::AbstractVertex, v2::AbstractVertex, visited=AbstractVertex[]) = false
function issame(v1::T, v2::T, visited=AbstractVertex[]) where T<:AbstractVertex
    v1 in visited && return true
    v2 in visited && return true
    push!(visited, v1,v2)
    for n in fieldnames(typeof(v1))
        issame(getfield(v1, n), getfield(v2, n), visited) || return false
    end
    return true
end
function issame(a1::AbstractArray{T,1}, a2::AbstractArray{T,1}, visited=AbstractVertex[]) where T<:AbstractVertex
    length(a1) != length(a2) && return false
    return all(map(vs -> issame(vs..., visited), zip(a1,a2)))
end
issame(g1::CompGraph, g2::CompGraph, visited=AbstractVertex[]) = issame(g1.outputs, g2.outputs, visited)
issame(d1, d2, visited=AbstractVertex[]) = d1 == d2

issame(s1::MutationOp, s2::MutationOp, visited=AbstractVertex[]) = false
function issame(s1::T, s2::T, visited=AbstractVertex[]) where T<:MutationOp
    for n in fieldnames(typeof(s1))
        issame(getfield(s1, n), getfield(s2, n), visited) || return false
    end
    return true
end

function showstr(f, v)
    buffer = IOBuffer()
    f(buffer, v)
    return String(take!(buffer))
end

# Testing mocks
mutable struct IndMem{F}
    wrapped::F
    lastins::Vector{Union{Missing, Vector{Int}}}
    lastouts::Vector{Int}
end
IndMem(w) = IndMem(w, convert(Vector{Union{Missing, Vector{Int}}}, map(i -> collect(1:i), nin(w))), collect(1:nout(w)))
IndMem(w, is::AbstractVector{<:Integer}, os::Integer) = IndMem(w, convert(Vector{Union{Missing, Vector{Int}}}, map(i -> collect(1:i), is)), collect(1:os))

NaiveNASlib.minΔninfactor(im::IndMem) = minΔninfactor(im.wrapped)
NaiveNASlib.minΔnoutfactor(im::IndMem) = minΔnoutfactor(im.wrapped)

(im::IndMem)(x...) = im.wrapped(x...)
NaiveNASlib.nout(im::IndMem) = nout(im.wrapped)
NaiveNASlib.nin(im::IndMem) = nin(im.wrapped)

lastins(v::AbstractVertex) = lastins(computation(v))
lastins(f) = missing
lastins(im::IndMem) = im.lastins

lastouts(v::AbstractVertex) = lastouts(computation(v))
lastouts(f) = missing
lastouts(im::IndMem) = im.lastouts

function NaiveNASlib.Δsize(im::IndMem, ins::AbstractVector, outs::AbstractVector)
    Δsize(im.wrapped, ins, outs)
    im.lastins = ins
    im.lastouts = outs
    nothing
end

mutable struct MatMul{M<:AbstractMatrix}
    W::M
end
MatMul(nin, nout) = MatMul(reshape(collect(1:nin*nout), nout,nin))

(mm::MatMul)(x) = mm.W * x

NaiveNASlib.minΔninfactor(::MatMul) = 1
NaiveNASlib.minΔnoutfactor(::MatMul) = 1

function NaiveNASlib.Δsize(mm::MatMul, ins::AbstractVector, outs::AbstractVector)
    mm.W = NaiveNASlib.parselect(mm.W, 1=>outs, 2=>ins[1])
    nothing
end
NaiveNASlib.nout(mm::MatMul) = size(mm.W, 1)
NaiveNASlib.nin(mm::MatMul) = [size(mm.W, 2)]

lastins(im::IndMem{<:MatMul}) = im.lastins[1]

computation(v::AbstractVertex) = computation(base(v))
computation(v::CompVertex) = v.computation
computation(::InputVertex) = identity