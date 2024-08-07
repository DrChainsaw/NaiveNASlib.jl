
"""
    base(v::AbstractVertex)

Return the vertex wrapped in `v` (if any).
"""
function base(::AbstractVertex) end

op(v::AbstractVertex) = op(base(v))

"""
    OutputsVertex

Decorates an AbstractVertex with output edges.
"""
struct OutputsVertex{V<:AbstractVertex} <: AbstractVertex
    base::V
    outs::Vector{AbstractVertex} # Untyped because we might add other vertices to it
end
OutputsVertex(v::AbstractVertex) = OutputsVertex(v, AbstractVertex[])
init!(v::OutputsVertex, p::AbstractVertex) = foreach(in -> push!(outputs(in), p), inputs(v))

base(v::OutputsVertex) = v.base
(v::OutputsVertex)(x...) = base(v)(x...)

inputs(v::OutputsVertex) = inputs(base(v))
outputs(v::OutputsVertex) = v.outs

# Must not copy the outs field as it is typically added by higher vertices
Functors.functor(::Type{<:OutputsVertex}, v) = (base = v.base,), newbase -> OutputsVertex(newbase[1])

"""
    InputSizeVertex

Vertex with an (immutable) size.

Intended use is for wrapping an InputVertex in conjuntion with mutation
"""
struct InputSizeVertex{V<:AbstractVertex} <: AbstractVertex
    base::V
    size::Int

    function InputSizeVertex(b::V, size::Integer) where V <:OutputsVertex
        this = new{V}(b, Int(size))
        init!(b, this)
        return this
    end
end
InputSizeVertex(name, size::Integer) = InputSizeVertex(InputVertex(name), size)
InputSizeVertex(b::AbstractVertex, size::Integer) = InputSizeVertex(OutputsVertex(b), size)

base(v::InputSizeVertex)::AbstractVertex = v.base
(v::InputSizeVertex)(x...) = base(v)(x...)

inputs(v::InputSizeVertex) = inputs(base(v))
outputs(v::InputSizeVertex) = outputs(base(v))

@functor InputSizeVertex

function Base.show(io::IO, v::InputSizeVertex)
    print(io, "InputSizeVertex(")
    show(io, base(v))
    print(io, ", ", v.size, ')')
end

"""
    MutationTrait

Base type for traits relevant when mutating.
"""
abstract type MutationTrait end

"""
    MutationSizeTrait
Base type for mutation traits relevant to size
"""
abstract type MutationSizeTrait <: MutationTrait end

"""
    SizeTransparent
Base type for mutation traits which are transparent w.r.t size, i.e size changes propagate both forwards and backwards.

Tip: Use with [`FixedSizeTrait`](@ref) if the function has parameters which must be aligned with the input and output sizes.
"""
abstract type SizeTransparent <: MutationSizeTrait end
"""
    SizeStack
Transparent size trait type where inputs are stacked, i.e output size is the sum of all input sizes.
"""
struct SizeStack <: SizeTransparent end
"""
    SizeInvariant
Transparent size trait type where all input sizes must be equal to the output size, e.g. elementwise operations (including broadcasted).
"""
struct SizeInvariant <: SizeTransparent end
"""
    SizeAbsorb
Size trait type for which size changes are absorbed, i.e they do not propagate forward.

Note that size changes do propagate backward as changing the input size of a vertex requires that the output size of its input is also changed and vice versa.
"""
struct SizeAbsorb <: MutationSizeTrait end
"""
    Immutable

Trait for vertices which are immutable. Typically inputs and outputs as those are fixed to the surroundings (e.g a data set).
"""
struct Immutable <: MutationTrait end

"""
    trait(v)

Return the [`MutationTrait`](@ref) for a vertex `v`.
"""
trait(::AbstractVertex) = Immutable()

"""
    DecoratingTrait <: MutationTrait

Avbstract trait which wraps another trait. The wrapped trait of a [`DecoratingTrait`](@ref) `t` is accessible through [`base(t)`](@ref base(t::DecoratingTrait)).
"""
abstract type DecoratingTrait <: MutationTrait end

"""
    base(t::DecoratingTrait) 

Return the trait wrapped by `t`.
"""
base(t::DecoratingTrait) = t.base # Lets just guess if we end up here :)

"""
    NamedTrait <: DecoratingTrait
    NamedTrait(name, base)

Trait which attaches `name` to a vertex. Calling [`name(v)`](@ref) on a vertex with this trait returns `name`.   
"""
struct NamedTrait{S, T<:MutationTrait} <: DecoratingTrait
    name::S
    base::T
end
base(t::NamedTrait) = t.base

name(t::DecoratingTrait) = name(base(t))
name(t::NamedTrait) = t.name
name(::MutationTrait) = nothing

@functor NamedTrait

function Base.show(io::IO, t::NamedTrait) 
    print(io, "NamedTrait(")
    show(io, t.name)
    print(io, ", ")
    show(io, t.base)
    print(io, ')')
end

"""
    FixedSizeTrait <: DecoratingTrait 

Trait which indicates that a vertex is [`SizeTransparent`](@ref) while still having a fixed size.

This prevents NaiveNASlib from inferring the size from neighbouring vertices.

As an example, the function `x -> 2 .* x` accepts any size of `x`, while the function `x -> [1,2,3] .* x`
is [`SizeInvariant`](@ref) but has a fixed size of 3.

Note that `FixedSizeTrait` does not imply that the vertex can't change size.
"""
struct FixedSizeTrait{T<:SizeTransparent} <: DecoratingTrait
    base::T
end
base(t::FixedSizeTrait) = t.base

"""
    AfterΔSizeTrait <: DecoratingTrait
    AfterΔSizeTrait(strategy::S, base::T)

Calls `after_Δnin(strategy, v, Δs, ischanged)` and `after_Δnout(strategy, v, Δ, ischanged)` after a size change for the vertex `v` which this trait is attached to.
"""
struct AfterΔSizeTrait{S, T<:MutationTrait} <: DecoratingTrait
    strategy::S
    base::T
end
base(t::AfterΔSizeTrait) = t.base

@functor AfterΔSizeTrait


"""
    MutationVertex

Vertex which may be subject to mutation.

The member trait describes the nature of the vertex itself, for example if size changes
are absorbed (e.g changing an `nin x nout` matrix to an `nin - Δ x nout` matrix) or if they
propagate to neighbouring vertices (and if so, how).
"""
struct MutationVertex{V<:AbstractVertex, T<:MutationTrait} <: AbstractVertex
    base::V
    trait::T

    function MutationVertex(b::V, t::T) where {V <: OutputsVertex, T <: MutationTrait}
        this = new{V, T}(b, t)
        init!(b, this)
        return this
    end
end
MutationVertex(b::AbstractVertex, t::MutationTrait) = MutationVertex(OutputsVertex(b), t)

@functor MutationVertex

base(v::MutationVertex) = v.base
trait(v::MutationVertex) = v.trait
(v::MutationVertex)(x...) = base(v)(x...)

inputs(v::MutationVertex)  = inputs(base(v))
outputs(v::MutationVertex) = outputs(base(v))

# Stuff for displaying information about vertices

show_less(io::IO, v::InputSizeVertex) = show_less(io, base(v))
show_less(io::IO, v::OutputsVertex) = show_less(io, base(v))

show_less(io::IO, v::MutationVertex) = show_less(io, trait(v), v)
show_less(io::IO, t::DecoratingTrait, v::AbstractVertex) = show_less(io, base(t), v)
show_less(io::IO, t::NamedTrait, ::AbstractVertex) = print(io, t.name)
show_less(io::IO, ::MutationTrait, v::AbstractVertex) = show_less(io, base(v))

function Base.show(io::IO, v::OutputsVertex; close=')')
     show(io, base(v); close="")
     print(io, ", outputs=")
     show(io, outputs(v))
     print(io, close)
 end

 function Base.show(io::IO, v::MutationVertex; close=')')
    print(io, "MutationVertex(")
    show(io, base(v))
    print(io, ", ")
    show(io, trait(v))
    print(io, close)
 end

# Stuff for logging

name(v::InputSizeVertex) = name(base(v))
name(v::OutputsVertex) = name(base(v))
name(v::MutationVertex) = name(trait(v), v)
name(t::MutationTrait, ::V) where V = string(nameof(V),  "::", summary(t))
name(t::NamedTrait, v) = t.name
name(t::DecoratingTrait, v) = name(base(t), v)

nameorrepr(v) = repr(v)
nameorrepr(v::InputSizeVertex) = nameorrepr(base(v))
nameorrepr(v::InputVertex) = name(v)
nameorrepr(v::MutationVertex) = nameorrepr(trait(v), v)
nameorrepr(t::DecoratingTrait, v) = nameorrepr(base(t), v)
nameorrepr(t::NamedTrait, v) = t.name
nameorrepr(::MutationTrait, v) = repr(v)

issizemutable(v::AbstractVertex) = issizemutable(trait(v))
issizemutable(t::DecoratingTrait) = issizemutable(base(t))
issizemutable(::MutationSizeTrait) = true
issizemutable(::Immutable) = false