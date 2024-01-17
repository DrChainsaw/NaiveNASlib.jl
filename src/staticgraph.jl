
struct TypedInputVertex{LABEL} <: AbstractVertex end
inputs(::TypedInputVertex) = tuple()

struct ImmutableCompVertex{F, I<:Tuple, LABEL} <: AbstractVertex
    computation::F
    noutputs::Int
    inputs::I
end
ImmutableCompVertex{LABEL}(computation::F, noutputs::Int, inputs::I) where {F, I, LABEL} = ImmutableCompVertex{F, I, LABEL}(computation, noutputs, inputs)
 
function ImmutableCompVertex(v::AbstractVertex, seen=IdDict()) 
    get!(seen, v) do
        ImmutableCompVertex(base(v), length(outputs(v)), seen)
    end
end
ImmutableCompVertex(v::AbstractVertex, noutputs, seen) = ImmutableCompVertex(base(v), noutputs, seen) 
ImmutableCompVertex(::InputVertex, noutputs, seen) = TypedInputVertex{length(seen)}()
function ImmutableCompVertex(v::CompVertex, noutputs, seen::IdDict) 
    ins = ImmutableCompVertex.(Tuple(inputs(v)), Ref(seen))
    comp = striptostable(v.computation)
    ImmutableCompVertex{length(seen)}(comp, noutputs, ins)
end

inputs(v::ImmutableCompVertex) = v.inputs
outputs(v::ImmutableCompVertex) = v.outputs
(v::ImmutableCompVertex)(x...) = v.computation(x...)

import NaiveNASflux: LazyMutable, MutableLayer, ActivationContribution

striptostable(f) = f
## This should go in NaioveNASflux
#= function striptostable(lm::LazyMutable) 
    NaiveNASflux.forcemutation(lm)
    striptostable(lm.mutable)
end
striptostable(ml::MutableLayer) = ml.layer
striptostable(ac::ActivationContribution) = ActivationContribution(striptostable(ac.layer), ac.contribution, ac.method) =#

struct ImmutableCompGraph{I, O}
    inputs::I
    outputs::O
end
function ImmutableCompGraph(g::CompGraph)
    seen = IdDict()
    outs = length(outputs(g)) === 1 ? ImmutableCompVertex(only(outputs(g)), seen) : ImmutableCompVertex.(Tuple(outputs(g)), Ref(seen))
    ins = length(inputs(g)) == 1 ? seen[only(inputs(g))] : Tuple(map(v -> seen[v]), inputs(g))
    ImmutableCompGraph(ins, outs)
end

function (g::ImmutableCompGraph{<:TypedInputVertex, <:ImmutableCompVertex})(x)
    evalcompgraph(g, x)
end

output_with_memo(memo, v::ImmutableCompVertex) = get_or_compute(memo, v) do mmemo, vv
    mnew, ins = _calc_outs3(mmemo, inputs(vv))
    out = vv(ins...)
    vv.noutputs > 1 ? (_memoize(mnew, vv, out), out) : (mnew, out)
end

vertices(g::ImmutableCompGraph) = ancestors(g.outputs)
Base.getindex(g::ImmutableCompGraph, args...) = getindex(vertices(g), args...)

function compgraphexpr(::Type{ImmutableCompGraph{I, O}}, gname) where {I, O}
    vexpr = vertexexpr(O, Set(), Set())
    vname = name(O)
    res = quote $vname = $gname.outputs end
    append!(res.args, vexpr.args)
    res
end

name(::Type{ImmutableCompVertex{F,T,vname}}) where {F,T,vname} = Symbol(:v, vname)
name(::Type{TypedInputVertex{vname}}) where vname = Symbol(:v,  vname)

activationname(t::Type{<:ImmutableCompVertex}) = Symbol(name(t), :_out)
activationname(t::Type{<:TypedInputVertex})  = Symbol(name(t), :_in)
 

function vertexexpr(t::Type{<:ImmutableCompVertex{F, T}}, seen, seeninputs) where {F, T<:Tuple{Vararg{Any}}}

    vname = name(t)
    vname in seen && return quote end
    push!(seen, vname)

    res = Expr(:block)
    resex = res.args
    ins = fieldtypes(T)
    invertexnames = name.(ins)
    invertices_to_assign = map(ivn -> ivn in seeninputs ? :_ : ivn, invertexnames)
    if !all(ivn -> ivn === :_, invertices_to_assign)
        push!(resex, :(($(invertices_to_assign...),) = $vname.inputs))
    end
    inputnames = activationname.(ins)
    vname = name(t)

    for (i, ft) in enumerate(ins)
        push!(seeninputs, invertexnames[i])
        ex = vertexexpr(ft, seen, seeninputs)
        append!(resex, ex.args)
    end
    vout = activationname(t)
    push!(resex, :($vout = $(vname)($(inputnames...))))
    res
end

function vertexexpr(t::Type{<:TypedInputVertex}, seen, seeninputs) 
    push!(seen, name(t))
    return quote end
end

@generated function evalcompgraph(g::ImmutableCompGraph{I}, v0_in) where I <: TypedInputVertex
    compgraphexpr(g, :g)
end
