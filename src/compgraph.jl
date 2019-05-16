import LightGraphs:SimpleDiGraph

"""
    CompGraph

Basic graph for computation. While not strictly neccessary to compute anything,
it makes it easier to keep track of things.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> cv = (CompVertex(+, InputVertex(1), InputVertex(2)));

julia> CompGraph(inputs(cv), [CompVertex(x -> 3x, cv)])(2,3) # (2 + 3) * 3
15

julia> CompGraph(inputs(cv), [cv, CompVertex(x -> 3x, cv)])(2,3)
(5, 15)
```

"""
struct CompGraph
    inputs::Array{AbstractVertex,1}
    outputs::Array{AbstractVertex,1}
end

function (g::CompGraph)(x...) where T <:Integer
    @assert length(x) == length(g.inputs) "Must supply one input for each input vertex!"
    memo::Dict{AbstractVertex, Any} = Dict(zip(g.inputs, x))
    if length(g.outputs) == 1
        return output!(memo, g.outputs[1])
    end
    return Tuple(map(v -> output!(memo, v), g.outputs))
end

"""
    output!(memo::Dict{AbstractVertex, Any}, v::AbstractVertex)

Return the output from v given any input in memo by traversing the graph.
Intermediate results from all visited vertices will be stored in memo after
function exits.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> ivs = InputVertex.(1:2);

julia> cv = CompVertex(*, ivs...);

julia> results = Dict{AbstractVertex, Any}(zip(ivs, [2,3]));

julia> output!(results, CompVertex(-, cv, ivs[1]))
4
julia> results
Dict{AbstractVertex,Any} with 4 entries:
  InputVertex(2)                                => 3
  InputVertex(1)                                => 2
  CompVertex(*, InputVertex(1), InputVertex(2)) => 6
  CompVertex(-, CompVertex(*), InputVertex(1))  => 4
```
"""
function output!(memo::Dict{AbstractVertex, Any}, v::AbstractVertex)
    # Calculate outputs which are not already calculated
    return get!(memo, v) do
        inpt = map(iv -> output!(memo, iv), inputs(v))
        out = v(inpt...)
    end
end

"""
    SimpleDiGraph(g::CompGraph)

Return g as a SimpleDiGraph.
"""
LightGraphs.SimpleDiGraph(g::CompGraph) = SimpleDiGraph(mapfoldl(v -> flatten(v), (vs1, vs2) -> unique(vcat(vs1, vs2)), g.outputs))

"""
    SimpleDiGraph(v::AbstractVertex)

Return a SimpleDiGraph of all parents of v
"""
LightGraphs.SimpleDiGraph(v::AbstractVertex)= SimpleDiGraph(flatten(v))

"""
    SimpleDiGraph(vertices::AbstractArray{AbstractVertex,1})

Return a SimpleDiGraph of all given vertices
"""
function LightGraphs.SimpleDiGraph(vertices::AbstractArray{AbstractVertex,1})
    g = LightGraphs.SimpleDiGraph()
    add_vertices!(g, length(vertices))
    for (ind, v) in enumerate(vertices)
        foreach(iv -> add_edge!(g, findall(vv -> vv == iv, vertices)[1], ind), inputs(v))
    end
    return g
end

"""
    flatten(v::AbstractVertex)

Return an array of all parents of v
"""
function flatten(v::AbstractVertex, vertices::Array{AbstractVertex,1} = Array{AbstractVertex,1}())
    v in vertices && return vertices
    pushfirst!(vertices, v)
    foreach(iv -> flatten(iv, vertices), inputs(v))
    return vertices
end
