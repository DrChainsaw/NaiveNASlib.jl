
"""
    base(v::AbstractVertex)

Return base vertex
"""
function base end

"""
    OutputsVertex

Decorates an AbstractVertex with output edges.
"""
struct OutputsVertex <: AbstractVertex
    base::AbstractVertex
    outs::AbstractVector{AbstractVertex}
end
OutputsVertex(v::AbstractVertex) = OutputsVertex(v, AbstractVertex[])
init!(v::OutputsVertex, p::AbstractVertex) = foreach(in -> push!(outputs(in), p), inputs(v))
clone(v::OutputsVertex, ins::AbstractVertex...) = OutputsVertex(clone(base(v), ins...))

base(v::OutputsVertex) = v.base
(v::OutputsVertex)(x...) = base(v)(x...)

inputs(v::OutputsVertex) = inputs(base(v))
outputs(v::OutputsVertex) = v.outs

"""
    InputSizeVertex

Vertex with an (immutable) size.

Intended use is for wrapping an InputVertex in conjuntion with mutation
"""
struct InputSizeVertex <: AbstractVertex
    base::AbstractVertex
    size::Integer

    function InputSizeVertex(b::OutputsVertex, size::Integer)
        this = new(b, size)
        init!(b, this)
        return this
    end
end
InputSizeVertex(name, size::Integer) = InputSizeVertex(InputVertex(name), size)
InputSizeVertex(b::AbstractVertex, size::Integer) = InputSizeVertex(OutputsVertex(b), size)
clone(v::InputSizeVertex, ins::AbstractVertex...) = InputSizeVertex(clone(base(v), ins...), v.size)

base(v::InputSizeVertex)::AbstractVertex = v.base
(v::InputSizeVertex)(x...) = base(v)(x...)

inputs(v::InputSizeVertex) = inputs(base(v))
outputs(v::InputSizeVertex) = outputs(base(v))

"""
    MutationTrait

Base type for traits relevant when mutating.
"""
abstract type MutationTrait end
# For convenience as 99% of all traits are immutable. Don't forget to implement for stateful traits or else there will be pain!
clone(t::MutationTrait) = t

"""
    MutationSizeTrait
Base type for mutation traits relevant to size
"""
abstract type MutationSizeTrait <: MutationTrait end

"""
    SizeTransparent
Base type for mutation traits which are transparent w.r.t size, i.e size changes propagate both forwards and backwards.
"""
abstract type SizeTransparent <: MutationSizeTrait end
"""
    SizeStack
Transparent size trait type where inputs are stacked, i.e output size is the sum of all input sizes
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
trait(v::AbstractVertex) = Immutable()

abstract type DecoratingTrait <: MutationTrait end

struct NamedTrait <: DecoratingTrait
    base::MutationTrait
    name
end
base(t::NamedTrait) = t.base

struct SizeChangeLogger <: DecoratingTrait
    level::LogLevel
    infostr::InfoStr
    base::MutationTrait
end
SizeChangeLogger(base::MutationTrait) = SizeChangeLogger(FullInfoStr(), base)
SizeChangeLogger(infostr::InfoStr, base::MutationTrait) = SizeChangeLogger(Logging.Info, infostr, base)
base(t::SizeChangeLogger) = t.base
infostr(t::SizeChangeLogger, v::AbstractVertex) = infostr(t.infostr, v)

struct SizeChangeValidation <: DecoratingTrait
    base::MutationTrait
end
base(t::SizeChangeValidation) = t.base

"""
    MutationVertex

Vertex which may be subject to mutation.

Scope is mutations which affect the input and output sizes as such changes needs to be propagated to the neighbouring vertices.

The member op describes the type of mutation, e.g if individual inputs/outputs are to be pruned vs just changing the size without selecting any particular inputs/outputs.

The member trait describes the nature of the vertex itself, for example if size changes
are absorbed (e.g changing an nin x nout matrix to an nin - Δ x nout matrix) or if they
propagate to neighbouring vertices (and if so, how).
"""
struct MutationVertex <: AbstractVertex
    base::AbstractVertex
    op::MutationOp
    trait::MutationTrait

    function MutationVertex(b::OutputsVertex, s::MutationOp, t::MutationTrait)
        this = new(b, s, t)
        init!(b, this)
        return this
    end
end
MutationVertex(b::AbstractVertex, s::MutationOp, t::MutationTrait) = MutationVertex(OutputsVertex(b), s, t)

AbsorbVertex(b::AbstractVertex, s::MutationState, f::Function=identity) = MutationVertex(b, s, f(SizeAbsorb()))

StackingVertex(b::AbstractVertex, f::Function=identity) = StackingVertex(OutputsVertex(b), f)
StackingVertex(b::Union{OutputsVertex, MutationVertex}, f::Function=identity) = MutationVertex(b, IoSize(nout.(inputs(b)), sum(nout.(inputs(b)))), f(SizeStack()))

InvariantVertex(b::AbstractVertex, f::Function=identity) = InvariantVertex(b, NoOp(), f)
InvariantVertex(b::AbstractVertex, op::MutationOp, f::Function=identity) = MutationVertex(OutputsVertex(b), op, f(SizeInvariant()))

clone(v::MutationVertex, ins::AbstractVertex...; opfun=cloneop, traitfun=clonetrait) = MutationVertex(clone(base(v), ins...), opfun(v), traitfun(v))
cloneop(v::MutationVertex) = clone(op(v))
clonetrait(v::MutationVertex) = clone(trait(v))

base(v::MutationVertex) = v.base
op(v::MutationVertex) = v.op
trait(v::MutationVertex) = v.trait
(v::MutationVertex)(x...) = base(v)(x...)

inputs(v::MutationVertex)  = inputs(base(v))
outputs(v::MutationVertex) = outputs(base(v))

# Stuff for displaying information about vertices

show_less(io::IO, v::InputSizeVertex) = show_less(io, base(v))
show_less(io::IO, v::MutationVertex) = print(io, name(v))
show_less(io::IO, v::OutputsVertex) = show_less(io, base(v))

function show(io::IO, v::OutputsVertex)
     show(io, base(v))
     print(io, ", outputs=")
     show(io, outputs(v))
 end

# Stuff for logging

name(v::InputSizeVertex) = name(base(v))
name(v::OutputsVertex) = name(base(v))
name(v::MutationVertex) = name(trait(v), v)
name(t::MutationTrait, v) = summary(v) * "::" * summary(t)
name(t::NamedTrait, v) = t.name
name(t::DecoratingTrait, v) = name(base(t), v)

struct MutationTraitInfoStr <: InfoStr  end
struct MutationSizeTraitInfoStr <: InfoStr  end
struct NinInfoStr <: InfoStr  end
struct NoutInfoStr <: InfoStr  end
SizeInfoStr() = ComposedInfoStr(PrefixedInfoStr("nin=", BracketInfoStr(NinInfoStr())), PrefixedInfoStr("nout=", BracketInfoStr(NoutInfoStr())))

struct OutputsInfoStr <: InfoStr
    infostr::InfoStr
end
OutputsInfoStr() =BracketInfoStr(OutputsInfoStr(NameInfoStr()))

NameAndIOInfoStr() = push!(NameAndInputsInfoStr(), PrefixedInfoStr("outputs=", OutputsInfoStr()))

FullInfoStr() = push!(NameAndIOInfoStr(), SizeInfoStr(), MutationSizeTraitInfoStr())

infostr(::NinInfoStr, v::AbstractVertex) = "unknown"
infostr(::NoutInfoStr, v::AbstractVertex) = "unknown"
infostr(::OutputsInfoStr, v::AbstractVertex) = "unknown"

infostr(i::MutationTraitInfoStr, v::AbstractVertex) = infostr(i, trait(v))
infostr(::MutationTraitInfoStr, t::MutationTrait) = replace(string(t), "\"" => "")
infostr(i::MutationSizeTraitInfoStr, v::AbstractVertex) = infostr(i, trait(v))
infostr(i::MutationSizeTraitInfoStr, t::DecoratingTrait) = infostr(i, base(t))
infostr(::MutationSizeTraitInfoStr, t::MutationSizeTrait) = string(t)
infostr(::MutationSizeTraitInfoStr, t::Immutable) = string(t)
infostr(::NinInfoStr, v::MutationVertex) = join(string.(nin(v)), ", ")
infostr(::NinInfoStr, v::InputSizeVertex) = "N/A"
infostr(::NoutInfoStr, v::MutationVertex) = string(nout(v))
infostr(::NoutInfoStr, v::InputSizeVertex) = string(nout(v))
infostr(i::OutputsInfoStr, v::InputSizeVertex) = infostr(i, base(v))
infostr(i::OutputsInfoStr, v::MutationVertex) = infostr(i, base(v))
infostr(i::OutputsInfoStr, v::OutputsVertex) = join(infostr.(i.infostr, outputs(v)), ", ")
