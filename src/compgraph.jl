"""
    CompGraph
    CompGraph(input::AbstractVertex, output::AbstractVertex)
    CompGraph(input::AbstractVector{<:AbstractVertex}, output::AbstractVertex)
    CompGraph(input::AbstractVertex, output::AbstractVector{<:AbstractVertex})

Basic graph for computation. While not strictly neccessary to compute anything,
it makes it easier to keep track of things.

# Examples
```jldoctest
julia> using NaiveNASlib

julia> v1 = inputvertex("in1", 1) + inputvertex("in2", 1);

julia> v2 = invariantvertex(x -> 3x, v1);

julia> CompGraph(inputs(v1), v2)(2,3) # (2 + 3) * 3
15

julia> CompGraph(inputs(v1), [v1, v2])(2,3)
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
    return map(v -> output!(memo, v), Tuple(outputs(g)))
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
```jldoctest
julia> using NaiveNASlib, NaiveNASlib.Advanced, NaiveNASlib.Extend

julia> ivs = InputVertex.(1:2);

julia> v1 = CompVertex(*, ivs...);

julia> v2 = CompVertex(-, v1, ivs[1]);

julia> results = Dict{AbstractVertex, Any}(zip(ivs, [2,3]));

julia> output!(results, v2)
4
julia> Pair{AbstractVertex, Int}[v=>results[v] for v in ancestors(v2)]
4-element Vector{Pair{AbstractVertex, Int64}}:
                                         InputVertex(1) => 2
                                         InputVertex(2) => 3
 CompVertex(*, inputs=[InputVertex(1), InputVertex(2)]) => 6
  CompVertex(-, inputs=[CompVertex(*), InputVertex(1)]) => 4
```
"""
function output!(memo::AbstractDict{K,V}, v::AbstractVertex) where {K,V}
    # Calculate outputs which are not already calculated
    return get!(memo, v) do
        inpt = map(iv -> output!(memo, iv), inputs(v))
        v(inpt...)
    end::V
end

## rrule synopsis: This is pretty much all about avoiding the gradient of get!
## We do this by calling get! in the primal part of a rrule
## Consequence of this is that we also put the pullbacks in the memo so that
## they are always available as we recurse through the graph
## Now, since memo is preloaded with inputs we intercept the call to output!
## and add a dummy pullback to the existing intputs, then we proceed with the
## specialized output_rrule! which assumes there is a (y, pullback) tuple in
## memo for every vertex.
## Since we don't want to mend together pullbacks from multiple input vertices
## recursively we do the recursion in _output_rrule! which does not have a 
## rrule defined for it and thus we outsource the recursion and map-unzipping 
## to the AD

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(output!), memo, v)
    for (k,v) in memo
        memo[k] = (v, Δ -> ntuple(i -> NoTangent(), 3)) # TODO: Gradient of input, i.e the x in CompGraph(x)
    end
    return rrule_via_ad(config, output_rrule!, memo, v)
end

function output_rrule!(args...)
    throw(ArgumentError("output_rrule! must only be called when computing gradients!"))
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(output_rrule!), memo::AbstractDict{K,V}, v::AbstractVertex) where {K,V}
    get!(memo, v) do 
        rrule_via_ad(config, _output_rrule!, memo, v) 
    end::V
end

function _output_rrule!(memo, v)
    inpt = map(iv -> output_rrule!(memo, iv),  inputs(v))
    res = v(inpt...)
    return res
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
    findvertices(vname::AbstractString, g::CompGraph)

Return all vertices for which [`name(v)`](@ref) == vname`.
"""
findvertices(vname::AbstractString, g::CompGraph) = findvertices(v -> name(v) == vname, g)
"""
    findvertices(vpat::Regex, g::CompGraph)

Return all vertices for which `vpat` matches [`name(v)`](@ref).
"""
findvertices(vpat::Regex, g::CompGraph) = findvertices(v -> occursin(vpat, name(v)), g)

"""
    findvertices(predicate, g::CompGraph)

Return all vertices for which `predicate(v)` return `true`.
"""
findvertices(predicate, g::CompGraph) = filter(predicate, vertices(g))

"""
    ancestors(v::AbstractVertex)

Return an array of all ancestors of `v`, including `v` itself.

# Examples
```jldoctest
julia> using NaiveNASlib, NaiveNASlib.Advanced, NaiveNASlib.Extend

julia> ancestors(invariantvertex(+, inputvertex("in", 1)))
2-element Vector{AbstractVertex}:
 InputSizeVertex(InputVertex(in, outputs=[CompVertex(+)]), 1)
 MutationVertex(CompVertex(+, inputs=[in], outputs=[]), SizeInvariant())
```
"""
ancestors(v::AbstractVertex,args...) = collect_vertices_from(inputs, v, args...)

"""
    descendants(v::AbstractVertex)

Return an array of all descendants of `v`, including `v` itself.

# Examples
```jldoctest
julia> using NaiveNASlib, NaiveNASlib.Advanced, NaiveNASlib.Extend

julia> descendants(invariantvertex(+, inputvertex("in", 1)) |> inputs |> first)
2-element Vector{AbstractVertex}:
 MutationVertex(CompVertex(+, inputs=[in], outputs=[]), SizeInvariant())
 InputSizeVertex(InputVertex(in, outputs=[CompVertex(+)]), 1)
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