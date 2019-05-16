"""
    AbstractVertex
Vertex base type
"""
abstract type AbstractVertex end

"""
    inputs(v::AbstractVertex)

Return an Array of vertices which are input to the given vertex.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> inputs(CompVertex(identity, InputVertex(1)))
1-element Array{AbstractVertex,1}:
 InputVertex(1)
```
"""
function inputs end

# To avoid too verbose console output
function show(io::IO, vs::Array{AbstractVertex,1})
    for (i, v) in enumerate(vs)
        show_noinputs(io, v)
        i != length(vs) && print(io, ", ")
    end
end
show_noinputs(io::IO, v::AbstractVertex) = summary(io, v)

"""
    InputVertex

Acts as a source of data to the graph and therefore does not need
any input vertices to feed it

# Examples
```julia-repl
julia> using NaiveNASlib

julia> InputVertex(1)
InputVertex(1)

julia> InputVertex("input")
InputVertex("input")
```
"""
struct InputVertex <: AbstractVertex
    name
end
(v::InputVertex)(x...) = error("Missing input $(v.name) to graph!")
inputs(v::InputVertex)::Array{AbstractVertex,1} = []
show_noinputs(io::IO, v::InputVertex) = show(io, v)
"""
    CompVertex

Maps input from input vertices to output. Must have at least one input vertex

# Examples
```julia-repl
julia> using NaiveNASlib

julia> CompVertex(+, InputVertex(1), InputVertex(2))
CompVertex(+, AbstractVertex[InputVertex(1), InputVertex(2)])
```
"""
struct CompVertex <: AbstractVertex
    computation
    inputs::AbstractArray{AbstractVertex,1}

    function CompVertex(c, inputs::AbstractArray{AbstractVertex,1})
         @assert !isempty(inputs) "Must have inputs!"
         new(c, inputs)
     end
end
inputs(v::CompVertex) = v.inputs
function show_noinputs(io::IO, v::CompVertex)
    print(io, "CompVertex(")
    show(io, v.computation)
    print(io, ")")
end

CompVertex(c, inputs::AbstractVertex...) = CompVertex(c, AbstractVertex[inputs...])


"""
    CompVertex(x...)

Map x to output from computation.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> CompVertex(x -> 4x, InputVertex(1))(2)
8

julia>CompVertex(*, InputVertex(1), InputVertex(2))(2,3)
6
```
"""
(v::CompVertex)(x...) = v.computation(x...)
