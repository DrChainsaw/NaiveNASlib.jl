
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

Return an array of all vertices which terminate size changes (i.e does not propagate them) seen through the given direction
 (typically [`inputs`](@ref) or [`outputs`](@ref)). A vertex will be present once for each unique path through which its seen.

The `other` direction may be specified and will be traversed if a [`SizeInvariant`](@ref) vertex is encountered.

Will return the given vertex if it is terminating.

# Examples
```jldoctest
julia> using NaiveNASlib, NaiveNASlib.Advanced

julia> v1 = inputvertex("v1", 3);

julia> v2 = inputvertex("v2", 3);

julia> v3 = conc(v1,v2,v1,dims=1);

julia> name.(findterminating(v1, outputs, inputs))
1-element Vector{String}:
 "v1"

julia> name.(findterminating(v3, outputs, inputs))
Any[]

julia> name.(findterminating(v3, inputs, outputs))
3-element Vector{String}:
 "v1"
 "v2"
 "v1"

julia> v5 = v3 + inputvertex("v4", 9);

julia> name.(findterminating(v3, outputs, inputs))
1-element Vector{String}:
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

    """
        isinsizecycle(v)

    Return true if removing `v` leads to a size cycle.  

    A size cycle is when at least one vertex `vi` has the impossible size constraint that `nout(vi) == nout(vi) + nout(vj)` where `vj` is a vertex in the graph (possibly `vi`).
    
    This tends to happen when concatenation is followed by an elementwise operation and removing one input `vi` to the concatenation and replacing it with another vertex `vj` 
    which is also input to the elementwise operation (possibly through any number of size transparent operations). Since the sum of input sizes to the concatenation must be 
    equal to the input size of the elementwise operation and one out of several inputs to the concatenation must also have the same output size as the input to the 
    elementwise operation we are stuck with an impossible constraint.

    As the algorithm is a bit messy (which imo is a sign that it is incorrect), here is a rundown of the idea:

    1. From the queired vertex, trace the graph forwards (output direction) until we hit a non-`SizeTransparent` vertex. If we encounter a `SizeStack` vertex store 
    this vertex in a `SeenSizeStack` and use it as the current state. If we encounter a `SizeInvariant` when we have `SeenSizeStack` as the state, we will store the 
    vertex and the `SeenSizeStack` as the output. The result from this is thus the set of potentially problematic vertices. If there are no potentially problematic
    vertices there can not be a size cycle and we return false.

    2. As the posed question is "If I remove `v` and connect its inputs to its outputs, will there be a size cycle?" we determine if shortcutting `v` creates a size transparent
    path between any input ancestor to `v` and any of the problematic vertices. To avoid having to check all ancestors, we just examine the set which terminates any size 
    transparency using `findterminating`, knowing that if `v` and a problematic vertex share such a terminating ancestor, then shortcutting `v` creates a size transparent path
    to the problematic vertex which may not have been there before. If there are no problematic vertices which share a common terminating ancestor with `v` we return false.

    3. For each problematic vertex `vp` and associated set of terminating ancestors `VPa` which passes the check in 2 we now check if any of the vertices in `VPa` sees `vp` as
    problematic for the same reason as `v` does, meaning that they both pass through the same concatenation at least one time before hitting `vp`. If not, it basically means 
    that `v` is alone in a `noop` concatenation and there is no `vj` from the first paragraph to create an impossible constraint. If this applies to all problematic vertices
    we don't have a size cycle and return false.
    """
    function isinsizecycle(v)
        # Step 1: Find all potentially problematic vertices seen from v. Result is a Dict mapping SizeInvariant vertices to the set of SizeStack vertices passed through on the way
        problemvertices = Dict(findproblemvertices(v))

        isempty(problemvertices) && return false

        # Begin step 2. Get the set of size change terminating ancestors for v
        v_ancestors = mapreduce(vi -> findterminating(vi, inputs), vcat, inputs(v); init=AbstractVertex[])

        # We temporaily remove v from the graph without recreating any connections.
        # This prevents that any size absorbing vertices which are input to v show up as problemancestors below
        # which would obviously ruin the alg as they would trivially see any vertex in problemvertices as as being problematic
        # for the exact same reason as v does.
        undo_rm = remove_with_undo!(v)

        iscycle = any(keys(problemvertices)) do problemvertex
            # These are not really problematic, but they are the ancestors of a problematic vertex
            # Note that we have cut off `v` from the graph, so none of its inputs show up in problemancestors 
            # unless it connects to problemvertex through another path
            problemancestors = findterminating(problemvertex, inputs)

            # Step 2: There can't be any size ccyle if there is no transparent path
            # from vs terminating ancestors to one of problemvertexs terminating
            # ancestors. 
            !any(pa -> pa in v_ancestors, problemancestors) && return false

            any(problemancestors) do ancestor
                # Aaaand step 3: Check if any terminating ancestor of a `problemvertex` sees `problemvertex` as problematic for the same reason as `v` 
                ancestorproblemvertices = Dict(findproblemvertices(ancestor))

                # No intersect -> no problem
                any(intersect(keys(problemvertices), keys(ancestorproblemvertices))) do commonproblem
                    s1 = problemvertices[commonproblem]
                    s2 = ancestorproblemvertices[commonproblem]
                    # Finally check that there is at least one SizeStack which both `v` and a terminating ancestor of `problemvertex` (after `v` is cut of)
                    # passes through.
                    any(vx -> vx in s2.vs, s1.vs)
                end
            end
        end
        # Reconnect `v` to the graph throgh the undo function
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