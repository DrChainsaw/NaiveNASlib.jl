
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

If `get_prop(g, e, :direction)` is of type `Output` this means that `Δnout` of `e.dst` is called after processing `e.src`.

If `get_prop(g, e, :direction)` is of type `Input` this means that `Δnin` of `e.dst` is called after processing `e.src`.
"""
function ΔSizeGraph()
    g = MetaDiGraph(0, :size, -1)
    set_indexing_prop!(g, :vertex)
    return g
end

"""
    ΔninSizeGraph(v)

Return a `ΔSizeGraph` for the case when nin of `v` is changed, i.e when Δnin(v, Δ) is called.
"""
ΔninSizeGraph(v) = ΔSizeGraph(Input(), v)

"""
    ΔnoutSizeGraph(v)

Return a `ΔSizeGraph` for the case when nout of `v` is changed, i.e when Δnout(v, Δ) is called.
"""
ΔnoutSizeGraph(v) = ΔSizeGraph(Output(), v)

function ΔSizeGraph(d::Direction, v)
    g = ΔSizeGraph()
    set_prop!(g, :start, v => d)
    verts = all_in_Δsize_graph(trait(v), d, v, (g, v))
    return g
end

function all_in_Δsize_graph(v::AbstractVertex, d::Direction, (g,from)::Tuple{MetaDiGraph, AbstractVertex})
    has_edge(g, vertexind!(g, v), vertexind!(g, from)) && return g
    add_edge!(g, from, v, d) && return g
    all_in_Δsize_graph(trait(v), d, v, (g, v))
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
SizeDiGraph(cg::CompGraph) = SizeDiGraph(mapfoldl(v -> flatten(v), (vs1, vs2) -> unique(vcat(vs1, vs2)), cg.outputs))

"""
    SizeDiGraph(v::AbstractVertex)

Return a SizeDiGraph of all parents of v
"""
SizeDiGraph(v::AbstractVertex)= SizeDiGraph(flatten(v))

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
all_in_Δsize_graph(vs::AbstractDict{<:AbstractVertex}, d::Direction) where N = all_in_Δsize_graph(keys(vs), d)
all_in_Δsize_graph(vs::NTuple{N, Pair{<:AbstractVertex}}, d::Direction) where N = all_in_Δsize_graph(first.(vs), d)
all_in_Δsize_graph(vs, d::Direction) where N = unique(mapreduce(v -> all_in_Δsize_graph(v, d), vcat, vs; init=AbstractVertex[]))

function all_in_Δsize_graph(v::AbstractVertex, d::Direction, visited=[])
    (v, d) in visited && return visited
    push!(visited, (v, d))
    all_in_Δsize_graph(trait(v),d, v, visited)
    return unique(map(e -> e[1], visited))
end
function all_in_Δsize_graph(v::AbstractVertex, d::Both, visited=[])
    all_in_Δsize_graph(v, Input(), visited)
    foreach(vout -> all_in_Δsize_graph(vout, Input(), visited), outputs(v))
    return unique(map(e -> e[1], visited))
end

all_in_Δsize_graph(t::DecoratingTrait, d, v, visited) = all_in_Δsize_graph(base(t), d, v, visited)
function all_in_Δsize_graph(::Immutable, ::Direction, v, visited) end
all_in_Δsize_graph(::SizeAbsorb, d, v, visited) = foreach(vn -> all_in_Δsize_graph(vn, opposite(d), visited), neighbours(d, v))
function all_in_Δsize_graph(::SizeTransparent, d, v, visited)
    foreach(vin -> all_in_Δsize_graph(vin, Output(), visited), inputs(v))
    foreach(vout -> all_in_Δsize_graph(vout, Input(), visited), outputs(v))
end
