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

clone(v::AbstractVertex, ins::AbstractVertex...; opfun) = clone(v, ins...)

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
InputVertex(1)

julia> InputVertex("input")
InputVertex("input")
```
"""
struct InputVertex <: AbstractVertex
    name
end
clone(v::InputVertex, ins::AbstractVertex...) = isempty(ins) ? InputVertex(v.name) : error("Input vertex got inputs: $(ins)!")
inputs(v::InputVertex)::AbstractArray{AbstractVertex,1} = []
(v::InputVertex)(x...) = error("Missing input $(v.name) to graph!")

show_less(io::IO, v::InputVertex) = show_less(io, v, v.name)   
show_less(io::IO, v::InputVertex, name::String) = print(io, name)
show_less(io::IO, v::InputVertex, name) = print(io, "InputVertex($(v.name))")

"""
    CompVertex

Maps input from input vertices to output. Must have at least one input vertex

# Examples
```julia-repl
julia> using NaiveNASlib

julia> CompVertex(+, InputVertex(1), InputVertex(2))
CompVertex(+, [InputVertex(1), InputVertex(2)])

julia> CompVertex(x -> 4x, InputVertex(1))(2)
8

julia>CompVertex(*, InputVertex(1), InputVertex(2))(2,3)
6
```
"""
struct CompVertex <: AbstractVertex
    computation
    inputs::AbstractVector{AbstractVertex}
end
CompVertex(c, ins::AbstractVertex...) = CompVertex(c, collect(ins))
clone(v::CompVertex, ins::AbstractVertex...) = CompVertex(clone(v.computation), ins...)
inputs(v::CompVertex)::AbstractVector{AbstractVertex} = v.inputs
(v::CompVertex)(x...) = v.computation(x...)

function show_less(io::IO, v::CompVertex)
    summary(io, v)
    print(io, "(")
    show(io, v.computation)
    print(io, ")")
end

clone(c::Function) = deepcopy(c)

function show(io::IO, v::CompVertex)
    show_less(io, v)
    print(io, ", inputs=")
    show(io, inputs(v))
end
