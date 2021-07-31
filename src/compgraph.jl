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
struct CompGraph{I<:AbstractVector{<:AbstractVertex}, O<:AbstractVector{<:AbstractVertex}}
    inputs::I
    outputs::O
end
CompGraph(input::AbstractVertex, output::AbstractVertex) = CompGraph([input], [output])
CompGraph(input::AbstractVector{<:AbstractVertex}, output::AbstractVertex) = CompGraph(input, [output])
CompGraph(input::AbstractVertex, output::AbstractVector{<:AbstractVertex}) = CompGraph([input], output)

function (g::CompGraph)(x...)
    @assert length(x) == length(g.inputs) "Must supply one input for each input vertex!"
    memo::Dict{AbstractVertex, Any} = Dict(zip(g.inputs, x))
    if length(g.outputs) == 1
        return output!(memo, g.outputs[1])
    end
    return Tuple(map(v -> output!(memo, v), g.outputs))
end

inputs(g::CompGraph) = g.inputs
outputs(g::CompGraph) = g.outputs

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
  CompVertex(*, [InputVertex(1), InputVertex(2)], [CompVertex(-)]) => 6
  CompVertex(-, [CompVertex(*), InputVertex(1)], [])               => 4
  InputVertex(1, [CompVertex(*), CompVertex(-)])                   => 2
  InputVertex(2, [CompVertex(*)])                                  => 3
```
"""
function output!(memo::AbstractDict{AbstractVertex, Any}, v::AbstractVertex)
    # Calculate outputs which are not already calculated
    return get!(memo, v) do
        inpt = map(iv -> output!(memo, iv), inputs(v))
        out = v(inpt...)
    end
end

"""
    nv(g::CompGraph)

Return the total number of vertices in the graph.
"""
nv(g::CompGraph) = length(vertices(g))

"""
    ancestors(v::AbstractVertex)

Return an array of all ancestors of `v`, including `v` itself.

# Examples
```julia-repl
julia> ancestors(CompVertex(+, InputVertex.(1:2)...))
3-element Array{AbstractVertex,1}:
 InputVertex(1)
 InputVertex(2)
 CompVertex(+, [InputVertex(1), InputVertex(2)])
"""
ancestors(v::AbstractVertex,args...) = collect_vertices_from(inputs, v, args...)

"""
    descendants(v::AbstractVertex)

Return an array of all descendants of `v`, including `v` itself.

# Examples
```julia-repl
julia> descendants(CompVertex(+, InputVertex.(1:2)...))
3-element Array{AbstractVertex,1}:
 InputVertex(1)
 InputVertex(2)
 CompVertex(+, [InputVertex(1), InputVertex(2)])
"""
descendants(v::AbstractVertex,args...) = collect_vertices_from(outputs, v, args...)

function collect_vertices_from(f, v::AbstractVertex, vertices::Vector{AbstractVertex} = Vector{AbstractVertex}(), visited::Vector{AbstractVertex} = Vector{AbstractVertex}())
    v in vertices && return vertices
    if !(v in visited)
        push!(visited, v)
        foreach(iv -> collect_vertices_from(f, iv, vertices), f(v))
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

"""
    Base.copy(g::CompGraph, cf=clone)

Copies the given `CompGraph g` into a new instance with identical structure.

Argument `cf` may be used to alter the copy on any level, e.g to add new `MutationTraits`.

# Examples
```julia-repl
julia> ivs = InputVertex.(1:2);

julia> cv = CompVertex(+, ivs...);

julia> graph = CompGraph(ivs, [cv])
CompGraph([InputVertex(1), InputVertex(2)], [CompVertex(+)])

julia> gcopy = copy(graph)
CompGraph([InputVertex(1), InputVertex(2)], [CompVertex(+)])

julia> gcopy == graph
false
"""
function Base.copy(g::CompGraph, cf=clone)
    # Can't just have each vertex copy its inputs as a graph with multiple outputs
    # might end up with multiple copies of the same vertex

    # Instead, use the same recursion as when calculating output and store the copies
    # in the memo

    # Will contain mapping between vertex in g and its copy
    # We could initialize it with inputs, but CompGraph does not require that inputs
    # are of type InputVertex
    memo = Dict{AbstractVertex, AbstractVertex}()
    foreach(ov -> copy!(memo, ov, cf), g.outputs)
    return CompGraph(
    map(iv -> memo[iv], g.inputs),
    map(ov -> memo[ov], g.outputs)
    )
end
clone(x;cf=nothing) = deepcopy(x)

"""
    copy!(memo::Dict{AbstractVertex, AbstractVertex}, v::AbstractVertex, cf=clone)

Recursively copy the input parents of v, ensuring that each vertex gets exactly one copy.

Results will be stored in the provided dict as a mapping between original and copy.

Argument `cf` may be used to alter the copy on any level, e.g to add new `MutationTraits`.

# Examples
```julia-repl

julia> result = Dict{AbstractVertex, AbstractVertex}();

julia> NaiveNASlib.copy!(result, CompVertex(+, InputVertex.(1:2)...));

julia> result
Dict{AbstractVertex,AbstractVertex} with 3 entries:
  InputVertex(2)                                  => InputVertex(2)
  InputVertex(1)                                  => InputVertex(1)
  CompVertex(+, [InputVertex(1), InputVertex(2)]) => CompVertex(+, [InputVertex(1), InputVertex(2)])
"""
function copy!(memo::Dict{AbstractVertex, AbstractVertex}, v::AbstractVertex, cf=clone)
    return get!(memo, v) do
        # Recurse until inputs(v) is empty
        ins = map(iv -> copy!(memo, iv, cf), inputs(v))
        cf(v, ins...; cf=cf)
    end
end
