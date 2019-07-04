
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
    immutablevertex(computation, outsize::Integer, inputs::AbstractVertex...; mutation=IoChange, traitdecoration=identity)

Return an immutable computation type vertex.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> v = immutablevertex(x -> x * [1 2 3; 4 5 6], 3, inputvertex("input", 2));

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
immutablevertex(computation, outsize::Integer, inputs::AbstractVertex...; mutation=IoChange, traitdecoration=identity) = vertex(computation, outsize, traitdecoration(Immutable()), inputs..., mutation=mutation)

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
    invariantvertex(computation, input; mutation=IoChange, traitdecoration=identity)

Return a mutable computation type vertex which is size invariant, i.e nin == nout.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> v = invariantvertex(x -> 2 .* x, inputvertex("input", 2));

julia> nin(v)
1-element Array{Int64,1}:
 2

julia> nout(v)
2

julia> v([1 2])
1×2 Array{Int64,2}:
 2  4
```
"""
invariantvertex(computation, input; mutation=IoChange, traitdecoration=identity) = vertex(computation, nout(input), traitdecoration(SizeInvariant()), input, mutation=mutation)

"""
    conc(v::AbstractVertex, vs::AbstractVertex...; dims, mutation=IoChange, traitdecoration=identity)

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
```
"""
conc(v::AbstractVertex, vs::AbstractVertex...; dims, mutation=IoChange, traitdecoration=identity) = vertex((x...) -> cat(x..., dims=dims), nout(v) + sum(nout.(vs)), traitdecoration(SizeStack()), v, vs..., mutation=mutation)

struct VertexConf
    mutation
    traitdecoration
end
mutationconf(m) = VertexConf(m, identity)
traitconf(t) = VertexConf(IoChange, t)
VertexConf() = VertexConf(IoChange, identity)

function invariant_outsize(vs::AbstractVertex...)
    outsize = unique([nout.(vs)...])
    length(outsize) == 1 || throw(DimensionMismatch("Dimensions must match! Got $([nout.(vs)...])"))
    return outsize[]
end

# Common wiring for all elementwise operations
elemwise(op, conf::VertexConf, vs::AbstractVertex...) = vertex((x...) -> op.(x...), invariant_outsize(vs...), conf.traitdecoration(SizeInvariant()), vs..., mutation=conf.mutation)


"""
    >>(conf::VertexConf, v::AbstractVertex)

Return inputs as a tuple. Only used to enable the `conf >> v1 op v2 ...` syntax.

"""
Base.:>>(conf::VertexConf, v::AbstractVertex) = (conf, v)

"""
    +(v::AbstractVertex, vs::AbstractVertex...)

Return a mutable vertex which performs (broadcasted) elementwise addition of its inputs.

A `VertexConf` with functions to create `MutationState` and `MutationTrait` can be supplied through the `>>` operator.

# Examples

```julia-repl
julia> using NaiveNASlib

julia> v = inputvertex("in1", 2) + inputvertex("in2", 2) + inputvertex("in3" ,2);

julia> nin(v)
3-element Array{Int64,1}:
 2
 2
 2

julia> nout(v)
2

julia> v([1, 2], [3, 4], [5, 6])
2-element Array{Int64,1}:
  9
 12

julia> name(v)
"MutationVertex::SizeInvariant"

julia> typeof(op(v))
IoChange

julia> conf = VertexConf(IoSize, t -> NamedTrait(t, "v"));

julia> v = conf >> inputvertex("in1", 3) + inputvertex("in2", 3);

julia> name(v)
"v"

julia> typeof(op(v))
IoSize
```
"""
Base.:+((conf, v)::Tuple{VertexConf, <:AbstractVertex}, vs::AbstractVertex...) = elemwise(+, conf, v, vs...)
Base.:+(v::AbstractVertex, vs::AbstractVertex...) = +(VertexConf() >> v, vs...)

"""
    *(v::AbstractVertex, vs::AbstractVertex...)
    *(conf::VertexConf, v::AbstractVertex, vs::AbstractVertex...)

Return a mutable vertex which performs (broadcasted) elementwise multiplication of its inputs.

A `VertexConf` with functions to create `MutationState` and `MutationTrait` can be supplied through the `>>` operator.

# Examples

```julia-repl
julia> using NaiveNASlib

julia> v = inputvertex("in1", 2) * inputvertex("in2", 2) * inputvertex("in3" ,2);

julia> nin(v)
3-element Array{Int64,1}:
 2
 2
 2

julia> nout(v)
2

julia> v([1, 2], [3, 4], [5, 6])
2-element Array{Int64,1}:
 15
 48

julia> name(v)
"MutationVertex::SizeInvariant"

julia> typeof(op(v))
IoChange

julia> conf = VertexConf(IoSize, t -> NamedTrait(t, "v"));

julia> v = conf >> inputvertex("in1", 3) * inputvertex("in2", 3);

julia> name(v)
"v"

julia> typeof(op(v))
IoSize

```
"""
Base.:*((conf, v)::Tuple{VertexConf, <:AbstractVertex}, vs::AbstractVertex...) = elemwise(*, conf, v, vs...)
Base.:*(v::AbstractVertex, vs::AbstractVertex...) = *(VertexConf() >> v, vs...)

"""
    -(v1::AbstractVertex, v2::AbstractVertex)

Return a mutable vertex which performs (broadcasted) elementwise subtraction of its inputs.

A `VertexConf` with functions to create `MutationState` and `MutationTrait` can be supplied through the `>>` operator.

# Examples

```julia-repl
julia> using NaiveNASlib

julia> v = inputvertex("in1", 2) - inputvertex("in2", 2)

julia> nin(v)
2-element Array{Int64,1}:
 2
 2

julia> nout(v)
2

julia> v([1, 2], [3, 4])
2-element Array{Int64,1}:
 -2
 -2

julia> name(v)
"MutationVertex::SizeInvariant"

julia> typeof(op(v))
IoChange

julia> conf = VertexConf(IoSize, t -> NamedTrait(t, "v"));

julia> v = conf >> inputvertex("in1", 3) - inputvertex("in2", 3);

julia> name(v)
"v"

julia> typeof(op(v))
IoSize
```
"""
Base.:-((conf, v1)::Tuple{VertexConf, <:AbstractVertex}, v2::AbstractVertex) = elemwise(-, conf, v1, v2)
Base.:-(v1::AbstractVertex, v2::AbstractVertex) = -(VertexConf() >> v1, v2)


"""
    -(v::AbstractVertex)

Return a mutable vertex which performs elementwise negation of its input.

A `VertexConf` with functions to create `MutationState` and `MutationTrait` can be supplied through the `>>` operator. Due to operator precedence, this has to be done in the following order: `-(conf >> v)`

#Examples

```julia-repl
julia> using NaiveNASlib

julia> v = -inputvertex("in", 2);

julia> v([1,2])
2-element Array{Int64,1}:
 -1
 -2

julia> conf = VertexConf(IoSize, t -> NamedTrait(t, "v"));

julia> v = -(conf >> inputvertex("in", 2));

julia> name(v)
"v"

julia> typeof(op(v))
IoSize
```
"""
Base.:-((conf, v)::Tuple{VertexConf, <:AbstractVertex}) = elemwise(-, conf, v)
Base.:-(v::AbstractVertex) = -(VertexConf() >> v)

"""
    /(v1::AbstractVertex, v2::AbstractVertex)
    /(conf::VertexConf, v1::AbstractVertex, v2::AbstractVertex)

Return a mutable vertex which performs (broadcasted) elementwise division of its inputs.

A `VertexConf` with functions to create `MutationState` and `MutationTrait` can be supplied through the `>>` operator.

# Examples

```julia-repl
julia> using NaiveNASlib

julia> v = inputvertex("in1", 2) / inputvertex("in2", 2)

julia> nin(v)
2/element Array{Int64,1}:
 2
 2

julia> nout(v)
2

julia> v([6, 8], [2, 4])
2/element Array{Int64,1}:
 3
 2

julia> name(v)
"MutationVertex::SizeInvariant"

julia> typeof(op(v))
IoChange

julia> conf = VertexConf(IoSize, t -> NamedTrait(t, "v"));

julia> v = conf >> inputvertex("in1", 3) / inputvertex("in2", 3);

julia> name(v)
"v"

julia> typeof(op(v))
IoSize
```
"""
Base.:/((conf, v1)::Tuple{VertexConf, <:AbstractVertex}, v2::AbstractVertex) = elemwise(/, conf, v1, v2)
Base.:/(v1::AbstractVertex, v2::AbstractVertex) = /(VertexConf() >> v1, v2)
