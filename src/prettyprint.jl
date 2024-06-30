Base.show(g::CompGraph, args...; kwargs...) = show(stdout, g, args...; kwargs...)
function Base.show(io::IO, g::CompGraph, args...; kwargs...) 
    # Don't print the summary table if we are printing some iterable (as indicated by presence of :SHOWN_SET)
    haskey(io, :SHOWN_SET) && return print(io, "CompGraph(", nvertices(g)," vertices)")
    graphsummary(io, g, args...; title="CompGraph with graphsummary:", kwargs...)
end

"""
    graphsummary([io], graph, extracolumns...; [inputhl], [outputhl], kwargs...)

Prints a summary table of `graph` to `io` using `PrettyTables.pretty_table`.

Extra columns can be added to the table by providing any number of `extracolumns` which can be one of the following:
* a function (or any callable object) which takes a vertex as input and returns the column content
* a `Pair` where the first element is the column name and the other element is what previous bullet describes

The keyword arguments `inputhl` (default `crayon"fg:black bg:249"`) and `outputhl` (default `inputhl`) can be used
to set the highlighting of the inputs and outputs to `graph` respectively. If set to `nothing` no special highlighting
will be used.

All other keyword arguments are forwarded to `PrettyTables.pretty_table`. Note that this allows for overriding the
default formatting, alignment and highlighting.

!!! warning "API Stability" 
    While this function is part of the public API for natural reasons, the exact shape of its output shall not be considered stable.

    `Base.show` for `CompGraph`s just forwards all arguments and keyword arguments to this method. This might change in the future.

### Examples
```jldoctest
julia> using NaiveNASlib

julia> g = let 
            v1 = "v1" >> inputvertex("in1", 1) + inputvertex("in2", 1)
            v2 = invariantvertex("v2", sin, v1)
            v3 = conc("v3", v1, v2; dims=1) 
            CompGraph(inputs(v1), v3)
        end;

julia> graphsummary(g)
┌────────────────┬───────────┬────────────────┬───────────────────┐
│ Graph Position │ Vertex Nr │ Input Vertices │ Op                │
├────────────────┼───────────┼────────────────┼───────────────────┤
│ Input          │ 1         │                │                   │
│ Input          │ 2         │                │                   │
│ Hidden         │ 3         │ 1,2            │ + (element wise)  │
│ Hidden         │ 4         │ 3              │ sin               │
│ Output         │ 5         │ 3,4            │ cat(x..., dims=1) │
└────────────────┴───────────┴────────────────┴───────────────────┘

julia> graphsummary(g, name, "input sizes" => nin, "output sizes" => nout)
┌────────────────┬───────────┬────────────────┬───────────────────┬──────┬─────────────┬──────────────┐
│ Graph Position │ Vertex Nr │ Input Vertices │ Op                │ Name │ input sizes │ output sizes │
├────────────────┼───────────┼────────────────┼───────────────────┼──────┼─────────────┼──────────────┤
│ Input          │ 1         │                │                   │ in1  │             │ 1            │
│ Input          │ 2         │                │                   │ in2  │             │ 1            │
│ Hidden         │ 3         │ 1,2            │ + (element wise)  │ v1   │ 1,1         │ 1            │
│ Hidden         │ 4         │ 3              │ sin               │ v2   │ 1           │ 1            │
│ Output         │ 5         │ 3,4            │ cat(x..., dims=1) │ v3   │ 1,1         │ 2            │
└────────────────┴───────────┴────────────────┴───────────────────┴──────┴─────────────┴──────────────┘
```
"""
graphsummary(g::CompGraph, extracolumns...; kwargs...) = graphsummary(stdout, g, extracolumns...; kwargs...)
function graphsummary(io, g::CompGraph, extracolumns...; 
                        inputhl=PrettyTables.crayon"fg:black bg:249", 
                        outputhl=inputhl,
                        kwargs...)
    t = summarytable(g, extracolumns...)
    
    # Default formatting
    arraytostr = (x, args...) -> x isa AbstractVector ? join(x, ",") : isnothing(x) ? "" : x
    rowhighligts = PrettyTables.Highlighter(Returns(true), function(h, x, i, j)
        !isnothing(inputhl) &&  i <= length(inputs(g)) && return inputhl
        !isnothing(outputhl) && i > length(t[1]) - length(outputs(g)) && return outputhl
        # Kinda random to enable highlights on even rows if we have more than seven rows
        # If we do use it, we want it to start with default for the first hidden layer/vertex regardless of how 
        # many input vertices there are (assuming we use different formatting for input vertices).
        length(t[1]) > 7 && iseven(i - !isnothing(inputhl) * length(inputs(g))) && return PrettyTables.crayon"fg:white bold bg:dark_gray"
        PrettyTables.crayon"default"
    end)

    PrettyTables.pretty_table(io, t; 
                        show_subheader=false, 
                        formatters=arraytostr, 
                        highlighters = rowhighligts, 
                        alignment = :l,
                        kwargs...)
end

function summarytable(g::CompGraph, extracols...)
    vs = vertices(g)

    inds = sort(collect(eachindex(vs)); by = function(i)
        vs[i] in inputs(g) && return i - length(vs)
        vs[i] in outputs(g) && return i + length(vs)
        i
    end)

    vs_roworder = vs[inds] 

    NamedTuple((
        Symbol("Graph Position") => map(v -> v in inputs(g) ? :Input : v in outputs(g) ? :Output : :Hidden, vs_roworder),
        Symbol("Vertex Nr") => inds,
        Symbol("Input Vertices") => map(v -> something.(indexin(inputs(v), vs), -1), vs_roworder),
        :Op => op.(vs_roworder),
        map(c -> _createextracol(c, vs_roworder), extracols)...
    ))
end

_createextracol(f, vs) = Symbol(uppercasefirst(string(f))) => f.(vs)
_createextracol(p::Pair, vs) = Symbol(first(p)) => last(p).(vs) 

## Other stuff related to printing long arrays of numbers assuming patterns which often happen
## when mutating, typically long streaks of -1 and ascending integers.

compressed_string(x) = string(x)
struct RangeState
    start
    cnt
end
struct ConsecState
    val
    cnt
end
struct AnyState
    val
end

function form_state(prev, curr)
    Δ = curr - prev
    Δ == 0 && return ConsecState(prev, 2)
    Δ == 1 && return RangeState(prev, 1)
    return AnyState(curr)
end

# Is this.... FP?
increment(h0::RangeState, h1::RangeState, buffer) = RangeState(h0.start, h0.cnt+1)
increment(h0::ConsecState, h1::ConsecState, buffer) = ConsecState(h0.val, h0.cnt+1)
function increment(h0::AnyState, h1::AnyState, buffer)
    write(buffer, "$(h0.val), ")
    return h1
end

function compressed_string(a::AbstractVector)
    length(a) < 20 && return string(a)
    buffer = IOBuffer()
    write(buffer, "[")

    prev = a[1]
    hyp = AnyState(a[1])
    for curr in a[2:end]
        hyp = new_state(hyp, prev, curr, buffer)
        prev = curr
    end
    write_state(hyp, buffer, true)
    write(buffer, "]")
    return String(take!(buffer))
end

new_state(h, prev, curr, buffer) = new_state(h, form_state(prev, curr), buffer)
new_state(h0::T, h1::T, buffer) where T = increment(h0, h1, buffer)
function new_state(h0, h1, buffer)
    write_state(h0, buffer)
    return h1
end

function write_state(h::RangeState, buffer, last=false)
    if h.cnt > 3
        write(buffer, "$(h.start),…, $(h.start + h.cnt)")
    else
        write(buffer, join(string.(h.start:h.start+h.cnt), ", "))
    end
    if !last
        write(buffer, ", ")
    end
end

function write_state(h::ConsecState, buffer, last=false)
    if h.cnt > 3
        write(buffer, "$(h.val)×$(h.cnt)")
    else
        write(buffer, join(repeat([h.val], h.cnt), ", "))
    end
    if !last
        write(buffer, ", ")
    end
end
function write_state(h::AnyState, buffer, last=false)
    if last
        write(buffer, string(h.val))
    end
end
