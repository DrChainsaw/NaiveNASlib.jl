import Base.show
"""
    AbstractVertex
Vertex base type
"""
abstract type AbstractVertex end

Base.Broadcast.broadcastable(v::AbstractVertex) = Ref(v)

"""
    inputs(v::AbstractVertex)

Return an Array of vertices which are input to the given vertex.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> inputs(CompVertex(identity, InputVertex(1)))
1-element Array{AbstractVertex,1}:
 InputVertex(1, [CompVertex(identity)])
```
"""
function inputs end

"""
    outputs(v::AbstractVertex)

Return an Array of vertices for which the given vertex is input to.

# Examples
```julia-repl
julia> using NaiveNASlib

julia> iv = InputVertex(1);

julia> cv = (CompVertex(identity, iv));

julia> outputs(iv)
1-element Array{AbstractVertex,1}:
 CompVertex(identity, [InputVertex(1)], [])
```
"""
function outputs end

"""
    InputVertex

Acts as a source of data to the graph and therefore does not need
any input vertices to feed it

# Examples
```julia-repl
julia> using NaiveNASlib

julia> InputVertex(1)
InputVertex(1)

julia> InputVertex("input")
InputVertex("input")
```
"""
struct InputVertex{N} <: AbstractVertex
    name::N
end
clone(v::InputVertex, ins::AbstractVertex...;cf=clone) = isempty(ins) ? InputVertex(cf(v.name,cf=cf)) : error("Input vertex got inputs: $(ins)!")
inputs(::InputVertex)::AbstractArray{AbstractVertex,1} = []
(v::InputVertex)(x...) = error("Missing input $(v.name) to graph!")

"""
    CompVertex

Maps input from input vertices to output. Must have at least one input vertex

# Examples
```julia-repl
julia> using NaiveNASlib

julia> CompVertex(+, InputVertex(1), InputVertex(2))
CompVertex(+, [InputVertex(1), InputVertex(2)])

julia> CompVertex(x -> 4x, InputVertex(1))(2)
8

julia>CompVertex(*, InputVertex(1), InputVertex(2))(2,3)
6
```
"""
struct CompVertex{F} <: AbstractVertex
    computation::F
    inputs::Vector{AbstractVertex} # Untyped because we might add other vertices to it
end
CompVertex(c, ins::AbstractArray{<:AbstractVertex}) = CompVertex(c, collect(AbstractVertex, ins))
CompVertex(c, ins::AbstractVertex...) = CompVertex(c, collect(AbstractVertex, ins))
clone(v::CompVertex, ins::AbstractVertex...;cf=clone) = CompVertex(cf(v.computation, cf=cf), ins...)
inputs(v::CompVertex) = v.inputs
(v::CompVertex)(x...) = v.computation(x...)

## Stuff for displaying information about vertices

# To avoid too verbose console output
function Base.show(io::IO, vs::AbstractVector{<:AbstractVertex})
    print(io, "[")
    for (i, v) in enumerate(vs)
        show_less(io, v)
        i != length(vs) && print(io, ", ")
    end
    print(io, "]")
end
show_less(io::IO, v::AbstractVertex; close=')') = summary(io, v)
show_less(io::IO, v::InputVertex; close=')') = show_less(io, v, v.name; close)
show_less(io::IO, ::InputVertex, name::String; close=')') = print(io, name)
show_less(io::IO, ::InputVertex, name; close=')') = print(io, "InputVertex(", name, close)

function show_less(io::IO, v::CompVertex; close=')')
    print(io, "CompVertex(")
    show(io, v.computation)
    print(io, close)
end

Base.show(io::IO, v::InputVertex; close=')') = print(io, "InputVertex(", v.name, close)
function Base.show(io::IO, v::CompVertex; close=')')
    print(io, "CompVertex(")
    show(io, v.computation)
    print(io, ", inputs=")
    show(io, inputs(v))
    print(io, close)
end

# Stuff for logging

name(v::AbstractVertex) = string(nameof(typeof(v)))
name(v::InputVertex) = v.name

# Don't call the mental hospital about the stuff below just yet!
# Let me explain: Debugging this library can be quite confusing due to the recursive nature of how changes propagate. To lessen the pain, is important to be able to format the vertices into something readable which carries the relevant information. As this varies from case to case it is highly customizable what can be printed and what can not.

abstract type InfoStr end
Base.Broadcast.broadcastable(i::InfoStr) = Ref(i)
Base.show(io::IO, istr::T) where T<:InfoStr = print(io, T)
clone(i::InfoStr, cf=clone) = i

struct RawInfoStr <: InfoStr end
struct NameInfoStr <: InfoStr end
struct BracketInfoStr <: InfoStr
    infostr::InfoStr
end

struct InputsInfoStr <: InfoStr
    infostr::InfoStr
end
InputsInfoStr() = BracketInfoStr(InputsInfoStr(NameInfoStr()))

struct PrefixedInfoStr <: InfoStr
    prefix
    infostr::InfoStr
end

struct ComposedInfoStr <: InfoStr
    infostrs::AbstractVector{<:InfoStr}
end
ComposedInfoStr(infostrs...) = ComposedInfoStr(collect(infostrs))
NameAndInputsInfoStr() = ComposedInfoStr(NameInfoStr(), PrefixedInfoStr("inputs=",  InputsInfoStr()))

function Base.push!(i::ComposedInfoStr, items...)
     push!(i.infostrs, items...)
     return i
 end

infostr(::RawInfoStr, v::AbstractVertex) = replace(string(v), "\"" => "")
infostr(::NameInfoStr, v::AbstractVertex) = name(v)
infostr(i::BracketInfoStr, v::AbstractVertex) = "[" * infostr(i.infostr, v) * "]"
infostr(i::InputsInfoStr, v::AbstractVertex) = join(infostr.(i.infostr, inputs(v)), ", ")
infostr(i::PrefixedInfoStr, v::AbstractVertex) = i.prefix * infostr(i.infostr, v)
infostr(i::ComposedInfoStr, v::AbstractVertex) = join(infostr.(i.infostrs, v), ", ")
