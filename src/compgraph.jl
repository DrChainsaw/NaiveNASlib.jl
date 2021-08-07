"""
    CompGraph
    CompGraph(input::AbstractVertex, output::AbstractVertex)
    CompGraph(input::AbstractVector{<:AbstractVertex}, output::AbstractVertex)
    CompGraph(input::AbstractVertex, output::AbstractVector{<:AbstractVertex})

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
struct CompGraph{I<:AbstractVector{<:AbstractVertex}, O<:AbstractVector{<:AbstractVertex}}
    inputs::I
    outputs::O
end
CompGraph(input::AbstractVertex, output::AbstractVertex) = CompGraph([input], [output])
CompGraph(input::AbstractVector{<:AbstractVertex}, output::AbstractVertex) = CompGraph(input, [output])
CompGraph(input::AbstractVertex, output::AbstractVector{<:AbstractVertex}) = CompGraph([input], output)

@functor CompGraph

function (g::CompGraph)(x...)
    @assert length(x) == length(g.inputs) "Must supply one input for each input vertex!"
    memo = Dict{AbstractVertex, Any}(zip(inputs(g), x))
    if length(outputs(g)) == 1
        return output!(memo, first(outputs(g)))
    end
    return Tuple(map(v -> output!(memo, v), outputs(g)))
end

"""
    inputs(g::CompGraph) 

Return the inputs vertices of `g`.
"""
inputs(g::CompGraph) = g.inputs

"""
    outputs(g::CompGraph) 

Return the output vertices of `g`.
"""
outputs(g::CompGraph) = g.outputs

"""
    output!(memo::AbstractDict{K, V}, v::AbstractVertex) where {K,V}

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
  CompVertex(*, [InputVertex(1), InputVertex(2)], [CompVertex(-)]) => 6
  CompVertex(-, [CompVertex(*), InputVertex(1)], [])               => 4
  InputVertex(1, [CompVertex(*), CompVertex(-)])                   => 2
  InputVertex(2, [CompVertex(*)])                                  => 3
```
"""
function output!(memo::AbstractDict{K,V}, v::AbstractVertex) where {K,V}
    # Calculate outputs which are not already calculated
    return get!(memo, v) do
        inpt = map(iv -> output!(memo, iv), inputs(v))
        v(inpt...)
    end::V
end

"""
    nvertices(g::CompGraph)

Return the number of vertices in the graph.
"""
nvertices(g::CompGraph) = length(vertices(g))

Base.getindex(g::CompGraph, args...) = getindex(vertices(g), args...)
Base.firstindex(g::CompGraph) = firstindex(vertices(g))
Base.lastindex(g) = lastindex(vertices(g))

"""
    findvertices(g::CompGraph, vname::AbstractString)

Return all vertices for which [`name(v)`](@ref) == vname`.
"""
findvertices(g::CompGraph, vname::AbstractString) = filter(v -> name(v) == vname, vertices(g))
"""
    findvertices(g::CompGraph, vname::Regex)

Return all vertices for which `vpat` matches [`name(v)`](@ref).
"""
findvertices(g::CompGraph, vpat::Regex) = filter(v -> match(vpat, name(v)) !== nothing, vertices(g))

"""
    ancestors(v::AbstractVertex)

Return an array of all ancestors of `v`, including `v` itself.

# Examples
```julia-repl
julia> ancestors(invariantvertex(+, inputvertex("in", 1)))
2-element Vector{AbstractVertex}:
 InputSizeVertex{OutputsVertex}(InputVertex(in, outputs=[CompVertex(+)]), 1)
 MutationVertex(CompVertex(+, inputs=[in], outputs=[]), SizeInvariant())
```
"""
ancestors(v::AbstractVertex,args...) = collect_vertices_from(inputs, v, args...)

"""
    descendants(v::AbstractVertex)

Return an array of all descendants of `v`, including `v` itself.

# Examples
```julia-repl
julia> descendants(invariantvertex(+, inputvertex("in", 1)) |> inputs |> first)
2-element Vector{AbstractVertex}:
 MutationVertex(CompVertex(+, inputs=[in], outputs=[]), SizeInvariant())
 InputSizeVertex{OutputsVertex}(InputVertex(in, outputs=[CompVertex(+)]), 1)
```
"""
descendants(v::AbstractVertex,args...) = collect_vertices_from(outputs, v, args...)

function collect_vertices_from(f, v::AbstractVertex, vertices::Vector{AbstractVertex} = Vector{AbstractVertex}(), visited::Vector{AbstractVertex} = Vector{AbstractVertex}())
    v in vertices && return vertices
    if !(v in visited)
        push!(visited, v)
        fvs = f(v)
        fvs === nothing && return vertices
        foreach(fv -> collect_vertices_from(f, fv, vertices), fvs)
    end
    push!(vertices, v)
    return vertices
end

"""
    vertices(g::CompGraph)

Return an topologically sorted array of all vertices in the graph `g`.

# Examples
```julia-repl
julia> ins = InputVertex.(1:2);

julia> v1 = CompVertex(+, ins...);

julia> v2 = CompVertex(*, v1, ins[2]);

julia> graph = CompGraph(ins, v2);

julia> vertices(graph)
4-element Array{AbstractVertex,1}:
 InputVertex(1)
 InputVertex(2)
 CompVertex(+), inputs=[InputVertex(1), InputVertex(2)]
 CompVertex(*), inputs=[CompVertex(+), InputVertex(2)]
```
"""
vertices(g::CompGraph) = unique(mapfoldl(ancestors, vcat, g.outputs))