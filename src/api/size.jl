
"""
    nout(v)

Return the number of output neurons of vertex `v`. 

This is typically the number of rows/columns of a parameter `Matrix` for fully connected layers or the number of output channels
in a convolutional layer. 

Note that NaiveNASlib does not have the capability to figure this out by itself so this must be provided by the function wrapped
in the vertex. For [`SizeTransparent`](@ref) vertices, NaiveNASlib will default to computing `nout(v)` from the sizes of the inputs.

There are three ways to implement `nin` for a function/callable `f` of type `F`:

    1. nout(f::F)
    2. nout(st::ST, f) and st = shapetrait(f)
    3. nout(f::F, t::MutationTrait, v::AbstractVertex)

While `1` is the most straight forward, `2` can be useful of there are many different `f`s which happen to share a common method
for determining the size.
Option `3` is when the implementation might want to use other information from `v` or `t` and is left as an escape hatch.
"""
function nout end

"""
    nin(v)

Return the number of input neurons of vertex `v`. 

This is typically the number of rows/columns of a parameter `Matrix` for fully connected layers or the number of input channels
in a convolutional layer. 

Note that NaiveNASlib does not have the capability to figure this out by itself so this must be provided by the function wrapped
in the vertex. For [`SizeTransparent`](@ref) vertices, NaiveNASlib will default to computing `nin(v)` from the sizes of the inputs.

There are three ways to implement `nin` for a function/callable `f` of type `F`:

    1. nin(f::F)
    2. nin(st::ST, f) and st = shapetrait(f)
    3. nin(f::F, t::MutationTrait, v::AbstractVertex)

While `1` is the most straight forward, `2` can be useful of there are many different `f`s which happen to share a common method
for determining the size.
Option `3` is when the implementation might want to use other information from `v` or `t` and is left as an escape hatch.
"""
function nin end


"""
    Δsize!([utilityfun], [s::AbstractΔSizeStrategy], vs::AbstractVector{<:AbstractVertex})

Change size of (potentially) all vertices in `vs` according to the provided [`AbstractΔSizeStrategy`](@ref) (default [`DefaultJuMPΔSizeStrategy`](@ref)).

Argument `utilityfun` provides a vector `utility = utilityfun(vx)` for any vertex `vx` in the same graph as `v` where 
`utility[i] > utility[j]` indicates that output neuron index `i` shall be preferred over `j` for vertex `vx`. It may also provide a scalar which will be used as utility of all neurons of `vx`. If not provided, [`defaultutility(vx)`](@ref) will be used.
"""
Δsize!(vs::AbstractVector{<:AbstractVertex}) = Δsize!(defaultutility, vs)
Δsize!(utilityfun, vs::AbstractVector{<:AbstractVertex}) = Δsize!(utilityfun, DefaultJuMPΔSizeStrategy(), vs)
Δsize!(s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}) = Δsize!(Δsizetype(vs), s, vs)
Δsize!(utilityfun, s::AbstractΔSizeStrategy, vs::AbstractVector{<:AbstractVertex}) = Δsize!(utilityfun, Δsizetype(vs), s, vs)

"""
    Δsize!(s::AbstractΔSizeStrategy) 
    Δsize!(utilityfun, s::AbstractΔSizeStrategy)

Change size of (potentially) all vertices which `s` has a chance to impact the size of.

Argument `utilityfun` provides a vector `utility = utilityfun(vx)` for any vertex `vx` in the same graph as `v` where 
`utility[i] > utility[j]` indicates that output neuron index `i` shall be preferred over `j` for vertex `vx`. It may also provide a scalar which will be used as utility of all neurons of `vx`. If not provided, [`defaultutility(vx)`](@ref) will be used.
"""
Δsize!(s::AbstractΔSizeStrategy) = Δsize!(s, add_participants!(s))
Δsize!(utilityfun, s::AbstractΔSizeStrategy) = Δsize!(utilityfun, s, add_participants!(s))

"""
    Δsize!([utilityfun], [s::AbstractΔSizeStrategy], g::CompGraph)
    Δsize!([utilityfun], [s::AbstractΔSizeStrategy], v::AbstractVertex)

Change size of (potentially) all vertices of graph `g` (or graph to which `v` is connected) according to the provided 
[`AbstractΔSizeStrategy`](@ref) (default [`DefaultJuMPΔSizeStrategy`](@ref)).

Return true of operation was successful, false otherwise.

Argument `utilityfun` provides a vector `utility = utilityfun(vx)` for any vertex `vx` in the same graph as `v` where 
`utility[i] > utility[j]` indicates that output neuron index `i` shall be preferred over `j` for vertex `vx`. It may also provide a scalar which will be used as utility of all neurons of `vx`. If not provided, [`defaultutility(vx)`](@ref) will be used.
"""
Δsize!(g::CompGraph) = Δsize!(defaultutility, g::CompGraph)
Δsize!(utilityfun, g::CompGraph) = Δsize!(utilityfun, DefaultJuMPΔSizeStrategy(), g)
Δsize!(s::AbstractΔSizeStrategy, g::CompGraph) = Δsize!(defaultutility, s::AbstractΔSizeStrategy, g::CompGraph)
Δsize!(utilityfun, s::AbstractΔSizeStrategy, g::CompGraph) = Δsize!(utilityfun, s, vertices(g))

Δsize!(v::AbstractVertex) = Δsize!(defaultutility, v::AbstractVertex)
Δsize!(utilityfun, v::AbstractVertex) = Δsize!(utilityfun, DefaultJuMPΔSizeStrategy(), v)
Δsize!(s::AbstractΔSizeStrategy, v::AbstractVertex) =Δsize!(defaultutility, s::AbstractΔSizeStrategy, v::AbstractVertex) 
Δsize!(utilityfun, s::AbstractΔSizeStrategy, v::AbstractVertex) = Δsize!(utilityfun, s, all_in_graph(v))


"""
    Δnin!([utilityfun], v, Δ...)
    Δnin!([utilityfun], v1 => Δ1, v2 => Δ2,...)
    Δnin!([utilityfun], Δs::AbstractDict)

Change input size of all provided vertices with the associated `Δ` and make the appropriate changes to other vertices
so that the graph is aligned w.r.t activations. Return `true` if successful (`false` if not successful).

For `Δ`s provided as integers it must be possible to change the size by exactly `Δ` or else the attempt will be considered failed.
A failed attempt will be retried immediately in relaxed form where the wanted size changes are moved to the objective.
The relaxation means that input size might not change by exactly `Δ`. Use [`relaxed(Δ)`](@ref) to indicate that a size change is 
relaxed in the initial attempt. 

For vertices with more than one input, the size change must be expressed as a tuple with one element per input. 
Use `missing` to indicate that no special treatment is needed for an input. Both `missing` and [`relaxed`](@ref) can be
mixed freely inside and outside the tuples (see examples).

Note that the above constrain makes `Δnin!` much more cumbersome to use compared to [`Δnout!`](@ref) and in most cases
there are no direct advantages of using `Δnin!` over [`Δnout!`](@ref) as they both boil down to the same thing. 

Argument `utilityfun` provides a vector `utility = utilityfun(vx)` for any vertex `vx` in the same graph as `v` where 
`utility[i] > utility[j]` indicates that output neuron index `i` shall be preferred over `j` for vertex `vx`. It may also provide a scalar which will be used as utility of all neurons of `vx`. If not provided, [`defaultutility(vx)`](@ref) will be used.

Note that `Δnin!([utilityfun], args...)` is equivalent to  `Δsize!([utilityfun], ΔNin(args...))`.

$(generic_Δnin_docstring_examples("Δnin!"; header="```jldoctest", footer=""))
julia> Δnin!(v3, relaxed(3), missing, 2);

julia> Δnin!(v3 => (relaxed(3), missing, 2), v4 => relaxed(-2, 0)) do v
           randn(nout(v))
       end;
```
"""
function Δnin! end

"""
    Δnout!([utilityfun], v, Δ...)
    Δnout!([utilityfun], v1 => Δ1, v2 => Δ2,...)
    Δnout!([utilityfun], Δs::AbstractDict)

Change output size of all provided vertices with the associated `Δ` and make the appropriate changes to other vertices
so that the graph is aligned w.r.t activations. Return `true` if successful (`false` if not successful).
    
For `Δ`s provided as integers it must be possible to change the size by exactly `Δ` or else the attempt will be considered failed.
A failed attempt will be retried immediately in relaxed form where the wanted size changes are moved to the objective.
The relaxation means that output size might not change by exactly `Δ`. Use [`relaxed(Δ)`](@ref) to indicate that a size change is 
relaxed in the initial attempt. 

Argument `utilityfun` provides a vector `utility = utilityfun(vx)` for any vertex `vx` in the same graph as `v` where 
`utility[i] > utility[j]` indicates that output neuron index `i` shall be preferred over `j` for vertex `vx`. It may also provide a scalar which will be used as utility of all neurons of `vx`. If not provided, [`defaultutility(vx)`](@ref) will be used.

Note that `Δnout!([utilityfun], args...)` is equivalent to `Δsize!([utilityfun], ΔNout(args...))`.

$(generic_Δnout_docstring_examples("Δnout!"; header="```jldoctest", footer=""))
julia> Δnout!(v1, relaxed(2));

julia> Δnout!(v1 => relaxed(2), v2 => -1) do v
           randn(nout(v))
       end;
```
"""
function Δnout! end

# Just another name for Δsize with the corresponding direction
Δnin!(args::Union{Pair{<:AbstractVertex}, AbstractDict{<:AbstractVertex}}...) = Δsize!(ΔNin(args...))
Δnout!(args::Union{Pair{<:AbstractVertex}, AbstractDict{<:AbstractVertex}}...) = Δsize!(ΔNout(args...))
Δnin!(v::AbstractVertex, Δs...) = Δnin!(v => Δs)
Δnout!(v::AbstractVertex, Δ) = Δnout!(v => Δ)

Δnin!(f, args::Union{Pair{<:AbstractVertex}, AbstractDict{<:AbstractVertex}}...) = Δsize!(f, ΔNin(args...))
Δnout!(f, args::Union{Pair{<:AbstractVertex}, AbstractDict{<:AbstractVertex}}...) = Δsize!(f, ΔNout(args...))
Δnin!(f, v::AbstractVertex, Δs...) = Δnin!(f, v => Δs)
Δnout!(f, v::AbstractVertex, Δ) = Δnout!(f, v => Δ)
