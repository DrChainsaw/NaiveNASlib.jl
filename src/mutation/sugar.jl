
"""
    inputvertex(name, size)

Return an immutable input type vertex with the given `name` and `size`.

Typically used as "entry" point to a computation graph.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> inputvertex("input", 5)
InputSizeVertex(InputVertex("input"), outputs=[], 5)

```
"""
inputvertex(name, size) = InputSizeVertex(name, size)


"""
    vertex(computation, outsize::Integer, trait::MutationTrait, inputs::AbstractVertex...; mutation=IoChange)

Return a mutable computation type vertex.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> v = vertex(x -> 5x, 1, SizeInvariant(), inputvertex("input", 1));

julia> v(3)
15
```
"""
vertex(computation, outsize::Integer, trait::MutationTrait, inputs::AbstractVertex...; mutation=IoChange) = MutationVertex(CompVertex(computation, inputs...), mutation(collect(nout.(inputs)), outsize), trait)

"""
    absorbvertex(computation, outsize::Integer, inputs::AbstractVertex...; mutation=IoChange, traitdecoration=identity)

Return a mutable computation type vertex which absorbs size changes. Typical example of this is a neural network layer.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> v = absorbvertex(x -> x * [1 2 3; 4 5 6], 3, inputvertex("input", 2));

julia> nin(v)
1-element Array{Int64,1}:
 2

julia> nout(v)
3

julia> v([1 2])
1×3 Array{Int64,2}:
 9  12  15

```
"""
absorbvertex(computation, outsize::Integer, inputs::AbstractVertex...; mutation=IoChange, traitdecoration=identity) = vertex(computation, outsize, traitdecoration(SizeAbsorb()), inputs..., mutation=mutation)

"""
    conc(vs::AbstractVertex...; dims, mutation=IoChange, traitdecoration=identity)

Return a mutable vertex which concatenates input along dimension `dim`.
# Examples
```julia-repl
julia> using NaiveNASlib

julia> v = conc(inputvertex.(["in1", "in2", "in3"], 1:3)..., dims=1);

julia> nin(v)
3-element Array{Int64,1}:
 1
 2
 3

julia> nout(v)
6

julia> v([1], [2, 3], [4, 5, 6])
6-element Array{Int64,1}:
 1
 2
 3
 4
 5
 6

"""
conc(vs::AbstractVertex...; dims, mutation=IoChange, traitdecoration=identity) = vertex((x...) -> cat(x..., dims=dims), sum(nout.(vs)), traitdecoration(SizeStack()), vs..., mutation=mutation)
