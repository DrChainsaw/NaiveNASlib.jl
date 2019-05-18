import Base.show
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
 InputVertex(1, [CompVertex(identity)])
```
"""
function inputs end

"""
    outputs(v::AbstractVertex)

Return an Array of vertices for which the given vertex is input to.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> iv = InputVertex(1);

julia> cv = (CompVertex(identity, iv));

julia> outputs(iv)
1-element Array{AbstractVertex,1}:
 CompVertex(identity, [InputVertex(1)], [])
```
"""
function outputs end

# To avoid too verbose console output
function Base.show(io::IO, vs::Array{AbstractVertex,1})
    print(io, "[")
    for (i, v) in enumerate(vs)
        show_less(io, v)
        i != length(vs) && print(io, ", ")
    end
    print(io, "]")
end
show_less(io::IO, v::AbstractVertex) = summary(io, v)

"""
    InputVertex

Acts as a source of data to the graph and therefore does not need
any input vertices to feed it

# Examples
```julia-repl
julia> using NaiveNASlib

julia> InputVertex(1)
InputVertex(1, [])

julia> InputVertex("input")
InputVertex("input", [])
```
"""
struct InputVertex <: AbstractVertex
    name
    outputs::AbstractArray{AbstractVertex,1}
end
InputVertex(name) = InputVertex(name, AbstractVertex[])
(v::InputVertex)(x...) = error("Missing input $(v.name) to graph!")
inputs(v::InputVertex)::AbstractArray{AbstractVertex,1} = []
outputs(v::InputVertex)::AbstractArray{AbstractVertex,1} = v.outputs
function show_less(io::IO, v::InputVertex)
    print(io, "InputVertex($(v.name))")
end

"""
    CompVertex

Maps input from input vertices to output. Must have at least one input vertex

# Examples
```julia-repl
julia> using NaiveNASlib

julia> CompVertex(+, InputVertex(1), InputVertex(2))
CompVertex(+, AbstractVertex[InputVertex(1), InputVertex(2)], [])

julia> CompVertex(x -> 4x, InputVertex(1))(2)
8

julia>CompVertex(*, InputVertex(1), InputVertex(2))(2,3)
6
```
"""
struct CompVertex <: AbstractVertex
    computation
    inputs::AbstractArray{AbstractVertex,1}
    outputs::AbstractArray{AbstractVertex,1}

    function CompVertex(c,
        ins::AbstractArray{AbstractVertex,1},
        outs::AbstractArray{AbstractVertex, 1})
        @assert !isempty(ins) "Must have inputs!"
        this = new(c, ins, outs)
        foreach(ins) do inpt
            push!(outputs(inpt), this)
        end
        return this
     end
end
inputs(v::CompVertex)::AbstractArray{AbstractVertex,1} = v.inputs
outputs(v::CompVertex)::AbstractArray{AbstractVertex,1} = v.outputs
function show_less(io::IO, v::CompVertex)
    print(io, "CompVertex(")
    show(io, v.computation)
    print(io, ")")
end

CompVertex(c, inputs::AbstractVertex...) = CompVertex(c, AbstractVertex[inputs...], AbstractVertex[])
(v::CompVertex)(x...) = v.computation(x...)
