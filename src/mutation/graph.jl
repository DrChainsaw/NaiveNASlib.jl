
# Vertex traits w.r.t whether size changes propagates
abstract type Direction end
"""
    Input

Represents the input direction, i.e coming from the output of another vertex.
"""
struct Input <: Direction end
"""
    Output

Represents the output direction, i.e coming from the input of another vertex.
"""
struct Output <: Direction end
"""
    Both

Represents both directions (`Input` and `Output`).
"""
struct Both <: Direction end

struct TightΔSizeGraph end
struct LooseΔSizeGraph end


"""
    opposite(d::Direction)

Return the opposite direction of `d`.
"""
opposite(::Input) = Output()
opposite(::Output) = Input()
opposite(b::Both) = b

"""
    neighbours(d::Direction, v)

Return vertices connected to `v` in direction `d`.
"""
neighbours(::Input, v) = inputs(v)
neighbours(::Output, v) = outputs(v)
neighbours(::Both, v) = vcat(inputs(v), outputs(v))


"""
    ΔSizeGraph

Represents the information on how a size change will propagate as a `MetaDiGraph`.

Each vertex `i` represents a unique `AbstractVertex vi`.
Each edge `e` represents propagation of a size change between vertices `e.src` and `e.dst`.
Edge weights denoted by the symbol `:size` represents the size of the output sent between the vertices.

For the `AbstractVertex vi` associated with vertex `i` in the graph `g`, the following holds `g[i, :vertex] == vi` and `g[vi,:vertex] == i`)

For an edge `e` in graph `g`, the following holds:

If `get_prop(g, e, :direction)` is of type `Output` this means that `Δnout!` of `e.dst` is called after processing `e.src`.

If `get_prop(g, e, :direction)` is of type `Input` this means that `Δnin!` of `e.dst` is called after processing `e.src`.
"""
function ΔSizeGraph()
    g = MetaDiGraph(0, :size, -1)
    set_indexing_prop!(g, :vertex)
    return g
end

"""
    ΔninSizeGraph(v)

Return a `ΔSizeGraph` for the case when nin of `v` is changed, i.e when Δnin!(v, Δ) is called.
"""
ΔninSizeGraph(v) = ΔSizeGraph(Input(), v)

"""
    ΔnoutSizeGraph(v)

Return a `ΔSizeGraph` for the case when nout of `v` is changed, i.e when Δnout!(v, Δ) is called.
"""
ΔnoutSizeGraph(v) = ΔSizeGraph(Output(), v)

function ΔSizeGraph(d::Direction, v)
    g = ΔSizeGraph()
    set_prop!(g, :start, v => d)
    verts = all_in_Δsize_graph(LooseΔSizeGraph(), trait(v), d, v, (g, v))
    return g
end

function all_in_Δsize_graph(mode, v::AbstractVertex, d::Direction, (g,from)::Tuple{MetaDiGraph, AbstractVertex})
    has_edge(g, vertexind!(g, v), vertexind!(g, from)) && return g
    add_edge!(g, from, v, d) && return g
    all_in_Δsize_graph(mode, trait(v), d, v, (g, v))
end

function LightGraphs.add_edge!(g::MetaDiGraph, src::AbstractVertex, dst::AbstractVertex, d::Direction)
    srcind = vertexind!(g, src)
    dstind = vertexind!(g, dst)
    visited(g, srcind, dstind, d) && return true
    add_edge!(g, srcind, dstind, Dict(:direction => d, g.weightfield => edgesize(d, src, dst)))
    return false
end

function visited(g, srcind, dstind, d)
    has_edge(g, srcind, dstind) && return true
    return visited_out(d, g, dstind)
end

function visited_out(::Output, g, dstind)
    for e in filter_edges(g, :direction, Output())
        e.dst == dstind && return true
    end
    return false
end
visited_out(::Input, g, dstind) = false

edgesize(::Input, src, dst) = nout(src)
edgesize(::Output, src, dst) = nout(dst)

vertexproplist(g::MetaDiGraph, prop::Symbol) = map(p -> p[:vertex], props.([g], vertices(g)))

function vertexind!(g::MetaDiGraph, v::AbstractVertex,)
    ind = indexin([v], vertexproplist(g, :vertex))[]
    if nothing == ind
        add_vertex!(g, :vertex, v)
    end
    return g[v, :vertex]
end


"""
    SizeDiGraph(cg::CompGraph)

Return `cg` as a `MetaDiGraph g`.

Each vertex `i` represents a unique `AbstractVertex vi`.

For the `AbstractVertex vi` associated with vertex `i` in the graph `g`, the following holds `g[i, :vertex] == vi` and `g[vi,:vertex] == i`)

Each edge `e` represents output from `e.src` which is input to `e.dst`.
Edge weights denoted by the symbol `:size` represents the size of the output sent between the vertices.

Note that the order of the input edges to a vertex matters (in case there are more than one) and is not encoded in `g`.
Instead, use `indexin(vi, inputs(v))` to find the index (or indices if input multiple times) of `vi` in `inputs(v)`.
"""
SizeDiGraph(cg::CompGraph) = SizeDiGraph(mapfoldl(v -> ancestors(v), (vs1, vs2) -> unique(vcat(vs1, vs2)), cg.outputs))

"""
    SizeDiGraph(v::AbstractVertex)

Return a SizeDiGraph of all parents of v
"""
SizeDiGraph(v::AbstractVertex)= SizeDiGraph(ancestors(v))

"""
    SizeDiGraph(vertices::AbstractArray{AbstractVertex,1})

Return a SizeDiGraph of all given vertices
"""
function SizeDiGraph(vertices::AbstractArray{<:AbstractVertex,1})
    g = MetaDiGraph(0,:size, 0)
    set_indexing_prop!(g, :vertex)
    add_vertices!(g, length(vertices))
    for (ind, v) in enumerate(vertices)
        set_prop!(g, ind, :vertex, v)
        for in_ind in indexin(inputs(v), vertices)
            add_edge!(g, in_ind, ind, :size, nout(vertices[in_ind]))
        end
    end
    return g
end

"""
    fullgraph(v::AbstractVertex)

Return a `SizeDiGraph` of all vertices in the same graph (or connected component) as `v`
"""
fullgraph(v::AbstractVertex) = SizeDiGraph(all_in_graph(v))

"""
    all_in_graph(v::AbstractVertex)

Return an array of vertices in the same graph (or connected component) as `v`
"""
function all_in_graph(v::AbstractVertex, visited = AbstractVertex[])
    v in visited && return visited
    push!(visited, v)
    foreach(vi -> all_in_graph(vi, visited), inputs(v))
    foreach(vo -> all_in_graph(vo, visited), outputs(v))
    return visited
end

"""
    all_in_Δsize_graph(v::AbstractVertex, d::Direction)

Return an array of vertices which will be affected if `v` changes size in direction `d`.
"""
all_in_Δsize_graph(vs::AbstractDict{<:AbstractVertex}, d::Direction, mode=LooseΔSizeGraph()) = all_in_Δsize_graph(keys(vs), d, mode)
all_in_Δsize_graph(vs::Tuple{Vararg{Pair{<:AbstractVertex}}}, d::Direction, mode=LooseΔSizeGraph()) = all_in_Δsize_graph(first.(vs), d, mode)
all_in_Δsize_graph(vs, d::Direction, mode=LooseΔSizeGraph()) = unique(mapreduce(v -> all_in_Δsize_graph(mode, v, d), vcat, vs; init=AbstractVertex[]))

all_in_Δsize_graph(v::AbstractVertex, d::Direction) = all_in_Δsize_graph(LooseΔSizeGraph(), v, d)

function all_in_Δsize_graph(mode, v::AbstractVertex, d::Direction, visited=[])
    (v, d) in visited && return visited
    push!(visited, (v, d))
    all_in_Δsize_graph(mode, trait(v), d, v, visited)
    return unique(map(e -> e[1], visited))
end
function all_in_Δsize_graph(mode, v::AbstractVertex, d::Both, visited=[])
    all_in_Δsize_graph(v, Input(), visited)
    foreach(vout -> all_in_Δsize_graph(mode, vout, Input(), visited), outputs(v))
    return unique(map(e -> e[1], visited))
end

all_in_Δsize_graph(mode, t::DecoratingTrait, d, v, visited) = all_in_Δsize_graph(mode, base(t), d, v, visited)
function all_in_Δsize_graph(mode, ::Immutable, ::Direction, v, visited) end
all_in_Δsize_graph(mode, ::SizeAbsorb, d, v, visited) = foreach(vn -> all_in_Δsize_graph(mode, vn, opposite(d), visited), neighbours(d, v))
function all_in_Δsize_graph(mode, ::SizeTransparent, d, v, visited)
    foreach(vin -> all_in_Δsize_graph(mode, vin, Output(), visited), inputs(v))
    foreach(vout -> all_in_Δsize_graph(mode, vout, Input(), visited), outputs(v))
end

function all_in_Δsize_graph(mode::TightΔSizeGraph, ::SizeStack, d::Input, v, visited) 
    if any(vx -> vx in inputs(v), first.(visited))
        push!(visited, (v, opposite(d)))
        foreach(vin -> all_in_Δsize_graph(mode, vin, d, visited), neighbours(opposite(d), v))
    else
        all_in_Δsize_graph(mode, SizeInvariant(), d, v, visited)
    end
end

"""
    findterminating(v::AbstractVertex, direction::Function, other::Function= v -> [], visited = [])

Return an array of all vertices which terminate size changes (i.e does not propagate them) seen through the given direction (typically inputs or outputs). A vertex will be present once for each unique path through which its seen.

The `other` direction may be specified and will be traversed if a SizeInvariant vertex is encountered.

Will return the given vertex if it is terminating.

# Examples
```julia-repl

julia> v1 = inputvertex("v1", 3);

julia> v2 = inputvertex("v2", 3);

julia> v3 = conc(v1,v2,v1,dims=1);

julia> name.(findterminating(v1, outputs, inputs))
1-element Array{String,1}:
 "v1"

julia> name.(findterminating(v3, outputs, inputs))
0-element Array{Any,1}

julia> name.(findterminating(v3, inputs, outputs))
3-element Array{String,1}:
 "v1"
 "v2"
 "v1"

 julia> v5 = v3 + inputvertex("v4", 9);

 julia> name.(findterminating(v3, outputs, inputs))
 1-element Array{String,1}:
  "v4"
```
"""
function findterminating(v::AbstractVertex, direction::Function, other::Function=v->AbstractVertex[], visited = Set{AbstractVertex}())
    v in visited && return AbstractVertex[]
    push!(visited, v)
    res = findterminating(trait(v), v, direction, other, visited)
    delete!(visited, v)
    return res
 end
findterminating(t::DecoratingTrait, v, d::Function, o::Function, visited) = findterminating(base(t), v, d, o, visited)
findterminating(::SizeAbsorb, v, d::Function, o::Function, visited) = [v]
findterminating(::Immutable, v, d::Function, o::Function, visited) = [v]

findterminating(::SizeStack, v, d::Function, o::Function, visited) = collectterminating(v, d, o, visited)
findterminating(::SizeInvariant, v, d::Function, o::Function, visited) = vcat(collectterminating(v, d, o, visited), collectterminating(v, o, d, visited))
collectterminating(v, d::Function, o::Function, visited) = mapfoldl(vf -> findterminating(vf, d, o, visited), vcat, d(v), init=[])

# Will be defined later but we need it below
function remove_with_undo! end

# Just a namespace so we don't have to see very generic names in the NaiveNASlib namespace as they are not to be used outside
module SizeCycleDetector

    export isinsizecycle

    using ..NaiveNASlib
    using ..NaiveNASlib:    AbstractVertex, remove_with_undo!, Input, Output, neighbours, DecoratingTrait, MutationTrait, 
                            SizeAbsorb, SizeStack, SizeInvariant, SizeTransparent, trait, base, findterminating

    struct SeenNothing end
    struct SeenSizeStack{T}
        vs::T
    end
    SeenSizeStack(v, s) = SeenSizeStack([v])
    SeenSizeStack(v, s::SeenSizeStack) = SeenSizeStack(vcat(v, s.vs))

    function isinsizecycle(v)
        problemvertices = Dict(findproblemvertices(v))

        isempty(problemvertices) && return false
        v_ancestors = mapreduce(vi -> findterminating(vi, inputs), vcat, inputs(v); init=AbstractVertex[])

        undo_rm = remove_with_undo!(v)

        iscycle = any(keys(problemvertices)) do problemvertex
            problemancestors = findterminating(problemvertex, inputs)

            # There can't be any size ccyle if there is no transparent path
            # from vs terminating ancestors to one of problemvertexs terminating
            # ancestors
            !any(pa -> pa in v_ancestors, problemancestors) && return false

            any(problemancestors) do ancestor

                ancestorproblemvertices = Dict(findproblemvertices(ancestor))

                any(intersect(keys(problemvertices), keys(ancestorproblemvertices))) do commonproblem
                    s1 = problemvertices[commonproblem]
                    s2 = ancestorproblemvertices[commonproblem]
                    any(vx -> vx in s2.vs, s1.vs)
                end
            end
        end

        undo_rm()
        return iscycle
    end

    findproblemvertices(v) = stepforward(SeenNothing(), v)
    findproblemvertices(state, v::AbstractVertex) = findproblemvertices(state, trait(v), v)
    findproblemvertices(state, t::DecoratingTrait, v) = findproblemvertices(state, base(t), v)
    findproblemvertices(state, ::MutationTrait, v) = pvinit()
    findproblemvertices(state::SeenNothing, ::SizeInvariant, v) = stepforward(state, v)
    findproblemvertices(state::SeenSizeStack, ::SizeInvariant, v) = vcat(v => state, stepforward(state, v))
    findproblemvertices(state, ::SizeStack, v) = stepforward(SeenSizeStack(v, state), v)

    stepforward(state, v) = mapreduce(vo -> findproblemvertices(state, vo), vcat, outputs(v); init=pvinit())

    pvinit() = Pair{AbstractVertex, <:SeenSizeStack}[]

end
using .SizeCycleDetector