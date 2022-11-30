"""
    AbstractVertex

Vertex base type.
"""
abstract type AbstractVertex end

Base.Broadcast.broadcastable(v::AbstractVertex) = Ref(v)
Functors.usecache(::Union{AbstractDict, AbstractSet}, ::AbstractVertex) = true

"""
    inputs(v)

Return an Array of vertices which are input to the given vertex.

# Examples
```jldoctest
julia> using NaiveNASlib, NaiveNASlib.Extend

julia> inputs(CompVertex(identity, InputVertex(1)))
1-element Vector{AbstractVertex}:
 InputVertex(1)
```
"""
function inputs(::AbstractVertex) end

"""
    outputs(v)

Return an Array of vertices for which the given vertex is input to.

# Examples
```jldoctest
julia> using NaiveNASlib

julia> iv = inputvertex("in", 3);

julia> cv = invariantvertex(identity, iv);

julia> outputs(iv)
1-element Vector{NaiveNASlib.AbstractVertex}:
 MutationVertex(CompVertex(identity, inputs=[in], outputs=[]), NaiveNASlib.SizeInvariant()) 
```
"""
function outputs(::AbstractVertex) end

"""
    InputVertex

Acts as a source of data to the graph and therefore does not need
any input vertices to feed it.

# Examples
```jldoctest
julia> using NaiveNASlib, NaiveNASlib.Extend

julia> InputVertex(1)
InputVertex(1)

julia> InputVertex("input")
InputVertex(input)
```
"""
struct InputVertex{N} <: AbstractVertex
    name::N
end
inputs(::InputVertex)::AbstractArray{AbstractVertex,1} = []
(v::InputVertex)(x...) = error("Missing input $(v.name) to graph!")

@functor InputVertex

"""
    CompVertex
    CompVertex(c, ins::AbstractVertex...)
    CompVertex(c, ins::AbstractArray{<:AbstractVertex}) =

Maps input from input vertices to output through `output = c(input...)`. 

Must have at least one input vertex.

# Examples
```jldoctest
julia> using NaiveNASlib, NaiveNASlib.Extend

julia> CompVertex(+, InputVertex(1), InputVertex(2))
CompVertex(+, inputs=[InputVertex(1), InputVertex(2)])

julia> CompVertex(x -> 4x, InputVertex(1))(2)
8

julia> CompVertex(*, InputVertex(1), InputVertex(2))(2,3)
6
```
"""
struct CompVertex{F} <: AbstractVertex
    computation::F
    inputs::Vector{AbstractVertex} # Untyped because we might add other vertices to it
end
CompVertex(c, ins::AbstractArray{<:AbstractVertex}) = CompVertex(c, collect(AbstractVertex, ins))
CompVertex(c, ins::AbstractVertex...) = CompVertex(c, collect(AbstractVertex, ins))
inputs(v::CompVertex) = v.inputs
(v::CompVertex)(x...) = v.computation(x...)

@functor CompVertex

## Stuff for displaying information about vertices

# To avoid too verbose console output
function Base.show(io::IO, vs::AbstractVector{<:AbstractVertex})
    print(io, "[")
    for (i, v) in enumerate(vs)
        show_less(io, v)
        i != length(vs) && print(io, ", ")
    end
    print(io, "]")
end
show_less(io::IO, v::AbstractVertex; close=')') = summary(io, v)
show_less(io::IO, v::InputVertex; close=')') = show_less(io, v, v.name; close)
show_less(io::IO, ::InputVertex, name::String; close=')') = print(io, name)
show_less(io::IO, ::InputVertex, name; close=')') = print(io, "InputVertex(", name, close)

function show_less(io::IO, v::CompVertex; close=')')
    print(io, "CompVertex(")
    show(io, v.computation)
    print(io, close)
end

Base.show(io::IO, v::InputVertex; close=')') = print(io, "InputVertex(", v.name, close)
function Base.show(io::IO, v::CompVertex; close=')')
    print(io, "CompVertex(")
    show(io, v.computation)
    print(io, ", inputs=")
    show(io, inputs(v))
    print(io, close)
end

# Stuff for logging

"""
    name(v)

Return a the name of the vertex `v`. 
Will return a generic string describing `v` if no name has been given to `v`. 

Note that names in a graph don't have to be unique. 
"""
name(v::AbstractVertex) = string(nameof(typeof(v)))
name(v::InputVertex) = v.name