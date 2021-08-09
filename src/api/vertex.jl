
"""
    inputvertex(name, size)

Return an immutable input type vertex with the given `name` and `size`.

Typically used as "entry" point to a computation graph.

# Examples
```jldoctest
julia> using NaiveNASlib

julia> inputvertex("input", 5)
InputSizeVertex(InputVertex(input, outputs=[]), 5)

```
"""
inputvertex(name, size) = InputSizeVertex(name, size)

gen_vertex_doc_sig(fname) = """
    $(fname)computation, inputs::AbstractVertex...; traitdecoration=identity)
    $(fname)vname::AbstractString, computation, inputs::AbstractVertex...; traitdecoration=identity)
    $(fname)computation, vname::AbstractString, inputs::AbstractVertex...; traitdecoration=identity)
"""

"""
$(gen_vertex_doc_sig("vertex(trait::MutationTrait, "))

Return a mutable computation type vertex.

# Examples
```jldoctest
julia> using NaiveNASlib

julia> v = NaiveNASlib.vertex(NaiveNASlib.SizeInvariant(), x -> 5x, inputvertex("input", 1));

julia> v(3)
15
```
"""
vertex(trait::MutationTrait, computation, inputs::AbstractVertex...) = MutationVertex(CompVertex(computation, inputs...), trait)
function vertex(trait::MutationTrait, vname::AbstractString, computation, inputs::AbstractVertex...) 
    othername = name(trait)
    if othername !== nothing
        @warn "Attach name $vname to vertex which already has name trait $(othername)!"
    end
    vertex(NamedTrait(vname, trait), computation, inputs...)
end
vertex(trait::MutationTrait, computation, vname::AbstractString, inputs::AbstractVertex...) = vertex(trait, vname, computation, inputs...)

"""
$(gen_vertex_doc_sig("immutablevertex("))

Return an immutable computation type vertex.

Use `traitdecoration` to attach other traits, such as [`named`](@ref), [`logged`](@ref) or [`validated`](@ref).

# Examples
```jldoctest
julia> using NaiveNASlib

julia> v = immutablevertex(x -> x * [1 2 3; 4 5 6], inputvertex("input", 2));

julia> v([1 2])
1×3 Matrix{Int64}:
 9  12  15

julia> v = immutablevertex("v", x -> x * [1 2 3; 4 5 6], inputvertex("input", 2));

julia> name(v)
"v"

julia> v = immutablevertex("v", inputvertex("input", 2)) do x
           x * [1 2 3; 4 5 6]
       end;

julia> name(v)
"v"       
```
"""
immutablevertex(args...; traitdecoration=identity) = vertex(traitdecoration(Immutable()), args...)

"""
$(gen_vertex_doc_sig("absorbvertex("))

Return a mutable computation type vertex which absorbs size changes. Typical example of this is a neural network layer
wrapping a parameter array.

Use `traitdecoration` to attach other traits, such as [`named`](@ref), [`logged`](@ref) or [`validated`](@ref).

# Examples
```jldoctest
julia> using NaiveNASlib

julia> v = absorbvertex(x -> x * [1 2 3; 4 5 6], inputvertex("input", 2));

julia> v([1 2])
1×3 Matrix{Int64}:
 9  12  15

julia> v = absorbvertex("v", x -> x * [1 2 3; 4 5 6], inputvertex("input", 2));

julia> name(v)
"v"

julia> v = absorbvertex("v", inputvertex("input", 2)) do x
           x * [1 2 3; 4 5 6]
       end;

julia> name(v)
"v"       
```
"""
absorbvertex(args...; traitdecoration=identity) = vertex(traitdecoration(SizeAbsorb()), args...)

"""
$(gen_vertex_doc_sig("invariantvertex("))

Return a mutable computation type vertex which is size invariant, i.e `nin == nout`.

Use `traitdecoration` to attach other traits, such as [`named`](@ref), [`logged`](@ref) or [`validated`](@ref).

# Examples
```jldoctest
julia> using NaiveNASlib

julia> v = invariantvertex(x -> 2 .* x, inputvertex("input", 2));

julia> nin(v)
1-element Vector{Int64}:
 2

julia> nout(v)
2

julia> v([1 2])
1×2 Matrix{Int64}:
 2  4

julia> v = invariantvertex("v", x -> 2 .* x, inputvertex("input", 2));

julia> name(v)
"v"

julia> v = invariantvertex("v", inputvertex("input", 2)) do x
           2 .* x
       end;

julia> name(v)
"v"    
```
"""
invariantvertex(args...; traitdecoration=identity) = vertex(traitdecoration(SizeInvariant()), args...)

"""
    conc(v::AbstractVertex, vs::AbstractVertex...; dims, traitdecoration=identity, outwrap=identity)
    conc(vname::AbstractString, v::AbstractVertex, vs::AbstractVertex...; dims, traitdecoration=identity, outwrap=identity)

Return a mutable vertex which concatenates input along dimension `dim`.

Use `traitdecoration` to attach other traits, such as [`named`](@ref), [`logged`](@ref) or [`validated`](@ref).

Use `outwrap=f` to wrap the concatenation function in `f`.

`conc(vname::AbstractString,...;traitdecoration = f, ...)` is equivalent to `conc(...;traitdecoration=named(vname) ∘ f, ...)`.

# Examples
```jldoctest
julia> using NaiveNASlib

julia> v = conc(inputvertex.(["in1", "in2", "in3"], 1:3)..., dims=1);

julia> nin(v)
3-element Vector{Int64}:
 1
 2
 3

julia> nout(v)
6

julia> v([1], [2, 3], [4, 5, 6])
6-element Vector{Int64}:
 1
 2
 3
 4
 5
 6

julia> v = conc(inputvertex.(["in1", "in2", "in3"], 1:3)..., dims=1, outwrap = f -> (x...) -> 2f(x...));
 
julia> v([1], [2, 3], [4, 5, 6])
6-element Vector{Int64}:
  2
  4
  6
  8
 10
 12
```
"""
function conc(v::AbstractVertex, vs::AbstractVertex...; dims, traitdecoration=identity, outwrap=identity)  
    vertex(traitdecoration(SizeStack()), outwrap((x...) -> cat(x..., dims=dims)), v, vs...)
end
function conc(vname::AbstractString, v::AbstractVertex, vs::AbstractVertex...; dims, traitdecoration=identity, outwrap=identity)
    vertex(traitdecoration(SizeStack()), vname, outwrap((x...) -> cat(x..., dims=dims)), v, vs...)
end


"""
    VertexConf
    VertexConf(; traitdecoration = identity, outwrap = identity)


Config struct to be used with element wise op syntax (`+`, `-`, `*`, `/`).

`traitdecoration` allows for decorating the vertex trait with stuff like logging, validation etc.
`outwrap` is a function which returns a function which will be applied to the computed output. 
For example, the following `outwrap` scales output by a factor of 2: `outwrap = f ->  (x...) -> 2f((x...)`
"""
struct VertexConf{TD, OW}
    traitdecoration::TD
    outwrap::OW
end
traitconf(t) = VertexConf(traitdecoration=t)
outwrapconf(o) = VertexConf(outwrap=o)
VertexConf(;traitdecoration = identity, outwrap = identity)= VertexConf(traitdecoration, outwrap)

# Common wiring for all elementwise operations
function elemwise(op, conf::VertexConf, vs::AbstractVertex...)
    all(vi -> nout(vi) == nout(vs[1]), vs) || throw(DimensionMismatch("nout of all vertices input to elementwise vertex must be equal! Got $(nout.(vs))"))
    invariantvertex(conf.outwrap((x...) -> op.(x...)), vs...; conf.traitdecoration)
end


"""
    >>(conf::VertexConf, v::AbstractVertex)
    >>(name::AbstractString, v::AbstractVertex)
    >>(outwrap::Function, v::AbstractVertex)

Return inputs as a tuple. Only used to enable the `conf >> v1 op v2 ...` syntax.

"""
Base.:>>(conf::VertexConf, v::AbstractVertex) = (conf, v)
Base.:>>(name::AbstractString, v::AbstractVertex) = traitconf(named(name)) >> v
Base.:>>(outwrap::Function, v::AbstractVertex) = outwrapconf(outwrap) >> v

"""
    +(v::AbstractVertex, vs::AbstractVertex...)

Return a mutable vertex which performs (broadcasted) elementwise addition of its inputs.

An `AbstractString`, `Function` or `VertexConf` can be supplied through the `>>` operator. 
An `AbstractString` will be used as the name of the vertex, a `Function f` will wrap the 
output so that the vertex computation becomes `f(+.(x...))` and `VertexConf` can be used
to supply both a name and a function.

# Examples

```jldoctest
julia> using NaiveNASlib

julia> v = inputvertex("in1", 2) + inputvertex("in2", 2) + inputvertex("in3" ,2);

julia> nin(v)
3-element Vector{Int64}:
 2
 2
 2

julia> nout(v)
2

julia> v([1, 2], [3, 4], [5, 6])
2-element Vector{Int64}:
  9
 12

julia> v = "v" >> inputvertex("in1", 3) + inputvertex("in2", 3);

julia> name(v)
"v"
```
"""
Base.:+((conf, v)::Tuple{VertexConf, <:AbstractVertex}, vs::AbstractVertex...) = elemwise(+, conf, v, vs...)
Base.:+(v::AbstractVertex, vs::AbstractVertex...) = +(VertexConf() >> v, vs...)

"""
    *(v::AbstractVertex, vs::AbstractVertex...)
    *(conf::VertexConf, v::AbstractVertex, vs::AbstractVertex...)

Return a mutable vertex which performs (broadcasted) elementwise multiplication of its inputs.

An `AbstractString`, `Function` or `VertexConf` can be supplied through the `>>` operator. 
An `AbstractString` will be used as the name of the vertex, a `Function f` will wrap the 
output so that the vertex computation becomes `f(*.(x...))` and `VertexConf` can be used
to supply both a name and a function.

# Examples

```jldoctest
julia> using NaiveNASlib

julia> v = inputvertex("in1", 2) * inputvertex("in2", 2) * inputvertex("in3" ,2);

julia> nin(v)
3-element Vector{Int64}:
 2
 2
 2

julia> nout(v)
2

julia> v([1, 2], [3, 4], [5, 6])
2-element Vector{Int64}:
 15
 48

julia> v = "v" >> inputvertex("in1", 3) * inputvertex("in2", 3);

julia> name(v)
"v"
```
"""
Base.:*((conf, v)::Tuple{VertexConf, <:AbstractVertex}, vs::AbstractVertex...) = elemwise(*, conf, v, vs...)
Base.:*(v::AbstractVertex, vs::AbstractVertex...) = *(VertexConf() >> v, vs...)

"""
    -(v1::AbstractVertex, v2::AbstractVertex)

Return a mutable vertex which performs (broadcasted) elementwise subtraction of its inputs.

An `AbstractString`, `Function` or `VertexConf` can be supplied through the `>>` operator. 
An `AbstractString` will be used as the name of the vertex, a `Function f` will wrap the 
output so that the vertex computation becomes `f(-.(x...))` and `VertexConf` can be used
to supply both a name and a function.

# Examples

```jldoctest
julia> using NaiveNASlib

julia> v = inputvertex("in1", 2) - inputvertex("in2", 2);

julia> nin(v)
2-element Vector{Int64}:
 2
 2

julia> nout(v)
2

julia> v([1, 2], [3, 4])
2-element Vector{Int64}:
 -2
 -2


julia> v = "v" >> inputvertex("in1", 3) - inputvertex("in2", 3);

julia> name(v)
"v"
```
"""
Base.:-((conf, v1)::Tuple{VertexConf, <:AbstractVertex}, v2::AbstractVertex) = elemwise(-, conf, v1, v2)
Base.:-(v1::AbstractVertex, v2::AbstractVertex) = -(VertexConf() >> v1, v2)


"""
    -(v::AbstractVertex)

Return a mutable vertex which performs elementwise negation of its input.

An `AbstractString`, `Function` or `VertexConf` can be supplied through the `>>` operator. 
An `AbstractString` will be used as the name of the vertex, a `Function f` will wrap the 
output so that the vertex computation becomes `f(-.(x...))` and `VertexConf` can be used
to supply both a name and a function.
Due to operator precedence, this has to be done in the following order: `-(conf >> v)`

#Examples

```jldoctest
julia> using NaiveNASlib

julia> v = -inputvertex("in", 2);

julia> v([1,2])
2-element Vector{Int64}:
 -1
 -2

julia> v = -("v" >> inputvertex("in", 2));

julia> name(v)
"v"
```
"""
Base.:-((conf, v)::Tuple{VertexConf, <:AbstractVertex}) = elemwise(-, conf, v)
Base.:-(v::AbstractVertex) = -(VertexConf() >> v)

"""
    /(v1::AbstractVertex, v2::AbstractVertex)
    /(conf::VertexConf, v1::AbstractVertex, v2::AbstractVertex)

Return a mutable vertex which performs (broadcasted) elementwise division of its inputs.

An `AbstractString`, `Function` or `VertexConf` can be supplied through the `>>` operator. 
An `AbstractString` will be used as the name of the vertex, a `Function f` will wrap the 
output so that the vertex computation becomes `f(/.(x...))` and `VertexConf` can be used
to supply both a name and a function.

# Examples

```jldoctest
julia> using NaiveNASlib

julia> v = inputvertex("in1", 2) / inputvertex("in2", 2);

julia> nin(v)
2-element Vector{Int64}:
 2
 2

julia> nout(v)
2

julia> v([6, 8], [2, 4])
2-element Vector{Float64}:
 3.0
 2.0

julia> v = "v" >> inputvertex("in1", 3) / inputvertex("in2", 3);

julia> name(v)
"v"
```
"""
Base.:/((conf, v1)::Tuple{VertexConf, <:AbstractVertex}, v2::AbstractVertex) = elemwise(/, conf, v1, v2)
Base.:/(v1::AbstractVertex, v2::AbstractVertex) = /(VertexConf() >> v1, v2)


"""
    named(name)

Return a function `t -> NamedTrait(name, t)` intended to reduce a bit of the verbosity when using `traitdecoration`.

Intended to be composable with other similar functions through ∘.
"""
named(name) = t -> NamedTrait(name, t)

"""
    validated(args...;kwargs...)

Return a function `t -> AfterΔSizeTrait(validateafterΔsize(args...;kwargs...), t)` indended to reduce a bit of the verbosity when using `traitdecoration`.

Intended to be composable with other similar functions through ∘.
"""
validated(args...;kwargs...) = t -> AfterΔSizeTrait(validateafterΔsize(args...;kwargs...), t)

"""
    logged(args...;kwargs...)

Return a function `t -> AfterΔSizeTrait(logafterΔsize(args...;kwargs...)` indended to reduce a bit of the verbosity when using `traitdecoration`.

Intended to be composable with other similar functions through ∘.
"""
logged(args...;kwargs...) = t -> AfterΔSizeTrait(logafterΔsize(args...;kwargs...), t)