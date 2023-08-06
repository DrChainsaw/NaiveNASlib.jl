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
struct CompGraph{I<:Union{Tuple, AbstractVertex}, O<:Union{Tuple, AbstractVertex}}
    inputs::I
    outputs::O
    function CompGraph(input, output)
        ivs = _vectotuple(input)
        ovs = _vectotuple(output)
        new{typeof(ivs), typeof(ovs)}(ivs, ovs)
    end
end
# The check is mostly to maintain legacy behaviour where a graph with a single output vertex always
# gave a non-tuple output
_vectotuple(x::AbstractVector) = length(x) === 1 ? x[1] : Tuple(x)
_vectotuple(x) = x

@functor CompGraph

function (g::CompGraph{<:AbstractVertex})(x)
    memo = init_memo(g.inputs, x)
    compute_graph(memo, g.outputs)
end

function (g::CompGraph{<:Tuple{Vararg{Any, N}}})(x::Vararg{Any, N}) where N
    memo = init_memo(g.inputs, x)
    compute_graph(memo, g.outputs)
end

(g::CompGraph)(x...) = throw(AssertionError("Must supply one input for each input vertex! Has $(length(g.inputs)) input vertices but got $(length(x)) inputs!"))

"""
    inputs(g::CompGraph) 

Return the inputs vertices of `g`.
"""
inputs(g::CompGraph{<:Tuple}) = collect(g.inputs)
inputs(g::CompGraph{<:AbstractVertex}) = [g.inputs]

"""
    outputs(g::CompGraph) 

Return the output vertices of `g`.
"""
outputs(g::CompGraph{<:Any, <:Tuple}) = collect(g.outputs)
outputs(g::CompGraph{<:Any, <:AbstractVertex}) = [g.outputs]

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

"""
    nvertices(g::CompGraph)

Return the number of vertices in the graph.
"""
nvertices(g::CompGraph) = length(vertices(g))

Base.getindex(g::CompGraph, args...) = getindex(vertices(g), args...)
Base.firstindex(g::CompGraph) = firstindex(vertices(g))
Base.lastindex(g::CompGraph) = lastindex(vertices(g))

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
vertices(g::CompGraph{<:Any, <:Tuple}) = unique(mapfoldl(ancestors, vcat, outputs(g)))
vertices(g::CompGraph{<:Any, <:AbstractVertex}) = ancestors(g.outputs)

## Non-public stuff to compute the CompGraph in a Zygote (and hopefully generally reverse-AD friendly) manner

compute_graph(memo, v::AbstractVertex) = last(output_with_memo(memo, v))
compute_graph(memo, vs::Tuple) = last(_calc_outs(memo, vs))

# Memo structs are similar to Base.ImmutableDict but tailormade for the CompGraph case
# also experimented with just using a (untyped) Vector with key => value pairs and 
# that worked too and had similar performance (maybe a little bit worse).
# Just feels better to have somewhat type stable memoization
abstract type AbstractMemo end

struct Memo{VT, OT} <: AbstractMemo
    key::VT
    value::OT
end
Memo() = Memo(tuple(), tuple()) # Use this as a sentinel value for empty memo

init_memo(v::AbstractVertex, x) = Memo(v, x)
init_memo(ks, vs) = init_memo(Memo(first(ks), first(vs)), Base.tail(ks), Base.tail(vs))
init_memo(m, ks, vs) = isempty(ks) ? m : init_memo(_memoize(m, first(ks), first(vs)), Base.tail(ks), Base.tail(vs))
# CompGraphs can have zero inputs. Not useful in general, but shows up in some tests of ONNXNaiveNASflux
init_memo(::Tuple{}, ::Tuple{}) = Memo() 

memokey(m::Memo) = m.key
memovalue(m::Memo) = m.value

struct LinkedMemo{PT<:AbstractMemo, VM <: Memo} <: AbstractMemo
    next::PT
    this::VM
end
memokey(m::LinkedMemo) = memokey(m.this)
memovalue(m::LinkedMemo) = memovalue(m.this)

_memoize(::Memo{Tuple{}, Tuple{}}, v, o) = Memo(v, o)
_memoize(vm::AbstractMemo, v, o) = _memoize(vm, Memo(v, o))
_memoize(vm1::AbstractMemo, vm2::Memo) = LinkedMemo(vm1, vm2)

get_or_compute(f, m::AbstractMemo, key) = get_or_compute(f, m, key, m)

# Zygote seems to prefer generated functions over recursion and loops
function get_or_compute_expr(f, m::Type{<:LinkedMemo{PT}}, key, topmemo) where PT
    ex = quote
        memokey(m) === key && return topmemo, memovalue(m)
        m = m.next
    end
    append!(ex.args, get_or_compute_expr(f, PT, key, topmemo).args)
    return ex
end
function get_or_compute_expr(f, m::Type{<:Memo}, key, topmemo)
    quote
        memokey(m) === key && return topmemo, memovalue(m)
        f(topmemo, key)
    end
end

@generated function get_or_compute(f, m::AbstractMemo, key, topmemo)
    get_or_compute_expr(f, m, key, topmemo)
end

# Only used for show method below, so we don't care that it is slow
Base.pairs(m::Memo) = tuple(memokey(m) => memovalue(m))
Base.pairs(m::LinkedMemo) = Iterators.flatten((pairs(m.this), pairs(m.next)))

function Base.show(io::IO, m::AbstractMemo) 
    print(io, "Memo(")
    namearr = map(pairs(m)) do (k, v)
        k isa AbstractVertex && return string(name(k), " => ", typeof(v))
        string(k, " => ", typeof(v)) 
    end

    print(io, join(namearr, ", "))
    print(io, ")")
end

output_with_memo(memo, v::AbstractVertex) = get_or_compute(memo, v) do mmemo, vv
    mnew, ins = _calc_outs(mmemo, inputs(vv))
    out = vv(ins...)
    # Memoizing everything or having a method to dispatch on MutationVertex resulted in worse gradient performance
    (!isa(vv, MutationVertex) || length(outputs(vv)) > 1) ? (_memoize(mnew, vv, out), out) : (mnew, out)
end

function _calc_outs_expr(memoname, vsname, ::Type{<:Tuple{Vararg{Any, N}}}) where N
    outs = ntuple( i -> Symbol(:out_, i), Val(N))
    calcexpr = map(i -> :((mnew, $(outs[i])) = output_with_memo(mnew, $vsname[$i])), 1:N)
    quote
        mnew = $memoname
        $(calcexpr...)
        mnew, tuple($(outs...))
    end
end

_calc_outs(memo, vs::AbstractArray) = _calc_outs(memo, Tuple(vs))
# Again: Zygote greatly preferred the generated function here over recursive and loop versions
@generated function _calc_outs(memo, vs::Tuple{Vararg{Any, N}}) where N
    _calc_outs_expr(:memo, :vs, vs)
end
