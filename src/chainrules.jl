
const enable_explicit_gradients = Ref(false)

abstract type AbstractMemo end

struct Memo{VT, OT} <: AbstractMemo
    key::VT
    value::OT
end
Memo(p::Pair) = Memo(first(p), last(p))
Memo(itr) = foldl((m, e) -> _memoize(m, e...), Iterators.drop(itr, 1); init=Memo(first(itr)...))

#= function init_memo(ks, vs) 
    memo = Memo(first(ks), first(vs))
    for i in 2:length(ks)
        memo = _memoize(memo, ks[i], vs[i])
    end
    memo
end =#

init_memo(ks::AbstractArray, vs) = init_memo(Tuple(ks), vs)
init_memo(ks, vs) = init_memo(Memo(first(ks), first(vs)), Base.tail(ks), Base.tail(vs))
init_memo(m, ks, vs) = isempty(ks) ? m : init_memo(_memoize(m, first(ks), first(vs)), Base.tail(ks), Base.tail(vs))

init_memo2(ks, vs) = [k => v for (k,v) in zip(ks, vs)]

memokey(m::Memo) = m.key
memovalue(m::Memo) = m.value

struct LinkedMemo{PT<:AbstractMemo, VM <: Memo} <: AbstractMemo
    next::PT
    this::VM
end
memokey(m::LinkedMemo) = memokey(m.this)
memovalue(m::LinkedMemo) = memovalue(m.this)

_memoize(vm::AbstractMemo, v, o) = _memoize(vm, Memo(v, o))
_memoize(vm1::AbstractMemo, vm2::Memo) = LinkedMemo(vm1, vm2)
_memoize(vm1::Memo, vm2::Memo) = LinkedMemo(vm2, vm1)

get_or_compute(f, m::AbstractMemo, key) = get_or_compute(f, m, key, m)

function get_or_compute(f, m::LinkedMemo, key, topmemo)
    memokey(m) == key && return topmemo, memovalue(m)
    get_or_compute(f, m.next, key, topmemo)
end
function get_or_compute(f, m::Memo, key, topmemo)
    memokey(m) == key && return topmemo, memovalue(m)
    f(key)
end

Base.pairs(m::Memo) = tuple(memokey(m) => memovalue(m))
Base.pairs(m::LinkedMemo) = Iterators.flatten((pairs(m.next), pairs(m.this)))

function Base.show(io::IO, m::AbstractMemo) 
    print(io, "Memo(")
    namearr = map(pairs(m)) do (k, v)
        k isa AbstractVertex && return name(k) => v
        k => v 
    end

    print(io, join(namearr, ", "))
    print(io, ")")
end


function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(output!), memo, v)
    res, back = rrule_via_ad(config, output_rrule!, memo, v)
    return res[2], function pullback_output(d)
        back((ChainRulesCore.ZeroTangent(), d))
    end
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(output!), memo::AbstractDict, v)
    rrule_via_ad(config, output_rrule!, memo, v)
end

# Workaround for https://github.com/FluxML/Zygote.jl/issues/1111
# and https://github.com/FluxML/Zygote.jl/issues/1243
# Only purpose is to return NoTangent, so whole function can be deleted
# if/when issues are resolved. Done forget to delete enable_explicit_gradients too then!
output!(memo::AbstractMemo, v) = last(output_rrule!(memo, v))
output!(memo::AbstractArray, v) = last(output_rrule!(memo, v))
output_rrule!(args...) = _output_rrule!(args...)
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(output_rrule!), memo, v)
    res, back = rrule_via_ad(config, _output_rrule!, memo, v)
    return res, function pullback_output_rrule!(d)
        bres = back(d)
        return enable_explicit_gradients[] ? bres : (NoTangent(), NoTangent(), NoTangent())
    end
end

function _output_rrule!(memo, v::AbstractVertex)
#=     get_or_compute(memo, v) do m, vv
        mnew, inpt = foldl(inputs(vv); init=(m, tuple())) do (mi, res), iv
           mnew, output = output_rrule!(mi, iv) 
           mnew, (res..., output)
        end
        out = v(inpt...)
        _memoize(mnew, v, v(inpt...)), out
    end =#

#=     get_or_compute(memo, v) do vv
        infostr("Compute ",  name(vv), " memo ", memo)
        mnew, inpt = foldl(inputs(vv); init=nothing) do state, iv
            if isnothing(state)
                m1, output1 = output_rrule!(memo, iv) 
                return m1, [output1]
            end
            mi, outs = state
            mnew, output = output_rrule!(mi, iv) 
            mnew, vcat(outs, [output])
        end
        out = v(inpt...)
        infostr("After compute ", name(vv), " memo: ", mnew)
        _memoize(mnew, v, out), out
    end =#

#=     get_or_compute(memo, v) do vv
        infostr("Compute ",  name(vv), " memo ", memo)
        mnew = memo
        inpt = nothing
        for iv in inputs(vv) 
            mnew, output = output_rrule!(mnew, iv) 
            inpt = isnothing(inpt) ? [output] : vcat(inpt, [output])
        end
        out = v(inpt...)
        infostr("After compute ", name(vv), " memo: ", mnew)
        _memoize(mnew, v, out), out
    end =#

#=     get_or_compute(memo, v) do vv
        infostr("Compute ",  name(vv), " memo ", memo)
        mnew, outs = output_rrule!(memo, first(inputs(vv)))
        inpt = [outs]
        for iv in inputs(vv)[2:end]
            mnew, output = output_rrule!(mnew, iv) 
            inpt = vcat(inpt, [output])
        end
        out = v(inpt...)
        infostr("After compute ", name(vv), " memo: ", mnew)
        _memoize(mnew, v, out), out
    end =#
    
    get_or_compute(memo, v) do vv
        mnew, inpt = calc_outs(memo, inputs(vv))
        out = vv(inpt...)
        _memoize(mnew, vv, out), out
    end 
#= 
    get_or_compute(memo, v) do vv
        mref = Ref{Any}(memo)
        inpt = map(inputs(vv)) do iv
            mref[], o = output_rrule!(mref[], iv)
            o
        end
        out = v(inpt...)
        _memoize(mref[], v, v(inpt...)), out
    end  =#


#=     get_or_compute(memo, v) do m, vv
        mref = Ref{Any}(m)
        inpt = map(inputs(vv)) do iv
            mref[], o = output_rrule!(mref[], iv)
            o
        end
        out = v(inpt...)
        _memoize(m, v, v(inpt...)), out
    end =#
#= 
    get_or_compute(memo, v) do m, vv
        aa= accumulate(inputs(vv); init=(m, nothing)) do (mi, _), iv
           mnew, output = output_rrule!(mi, iv) 
        end
        out = v(inpt...)
        _memoize(mnew, v, v(inpt...)), out
    end =#

    # rrule for get! not implemented, so we need to check the dict twice
#=     v in keys(memo) && return memo[v]
    inpt = map(iv -> output_rrule!(memo, iv),  inputs(v))
    memo[v] = v(inpt...) =#
end

function calc_outs(memo, vs)
    mnew, out = output_rrule!(memo, vs[1])
    calc_outs(tuple(out), mnew, vs, 2)
end
function calc_outs(outs, memo, vs, ind)
    ind > length(vs) && return memo, outs
    mnew, out = output_rrule!(memo, vs[ind])
    calc_outs((outs..., out), mnew, vs, ind+1)
end

infostr(args...) = @info string(args...)
ChainRulesCore.@non_differentiable infostr(args...) 


function _output_rrule!(memo::Dict, v::AbstractVertex)
    # rrule for get! not implemented, so we need to check the dict twice
    v in keys(memo) && return memo[v]
    inpt = map(iv -> output_rrule!(memo, iv),  inputs(v))
    memo[v] = v(inpt...)
end


get_or_compute(f, m::AbstractDict, x) = if haskey(m, x)
    m, m[x]
else
    _, v = f(x)
    m[x] = v
    m, v
end

_memoize(d::Dict, k, v) = d

function get_or_compute(f, a::AbstractArray, key)
    for (k,v) in a
        if key === k
            return a, v 
        end
    end
    f(key)
end
_memoize(a::AbstractArray, k, v) = vcat(a, k => v)
