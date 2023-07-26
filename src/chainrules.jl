abstract type AbstractMemo end

struct Memo{VT, OT} <: AbstractMemo
    key::VT
    value::OT
end
Memo(p::Pair) = Memo(first(p), last(p))
Memo(itr) = foldl((m, e) -> _memoize(m, e...), Iterators.drop(itr, 1); init=Memo(first(itr)...))


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

get_or_compute(f, m::AbstractMemo, key) = get_or_compute(f, m, key, m)

function _get_or_compute(f, m::LinkedMemo, key, topmemo)
    memokey(m) === key && return (topmemo, memovalue(m))
    get_or_compute(f, m.next, key, topmemo)
end
function _get_or_compute(f, m::Memo, key, topmemo)
    memokey(m) === key && return (topmemo, memovalue(m))
    f(key)
end

function _get_or_compute2(f, m::AbstractMemo, key, topmemo)
    while !isa(m, Memo)
        memokey(m) === key && return topmemo, memovalue(m)
        m = m.next
    end
    memokey(m) === key && return topmemo, memovalue(m)
    f(key)
end

Base.pairs(m::Memo) = tuple(memokey(m) => memovalue(m))
Base.pairs(m::LinkedMemo) = Iterators.flatten((pairs(m.this), pairs(m.next)))

function Base.show(io::IO, m::AbstractMemo) 
    print(io, "Memo(")
    namearr = map(pairs(m)) do (k, v)
        k isa AbstractVertex && return name(k) => typeof(v)
        k => typeof(v) 
    end

    print(io, join(namearr, ", "))
    print(io, ")")
end

function get_or_compute(f, a::AbstractArray, key)
    for (k,v) in a
        if key === k
            return a, v 
        end
    end
    f(key)
end
_memoize(a::AbstractArray, k, v) = vcat(a, k => v)

struct MemoCallback{M}
    memo::M
end
function (m::MemoCallback)(v)
    mnew, out = calc_outs(v, m.memo, inputs(v))
    length(outputs(v)) > 1 ? (_memoize(mnew, v, out), out) : (mnew, out)
end

compute_graph(memo::AbstractMemo, v) = last(output_with_memo(memo, v))
compute_graph(memo::AbstractArray, v) = last(output_with_memo(memo, v))
_output_with_memo(memo, v::AbstractVertex) = get_or_compute(MemoCallback(memo), memo, v) 

# ChainRules/Zygote friendly combination of map and reduction (map over vs, reduction of memo)
function _calc_outs(f, memo, vs)
    mnew, in1 = output_with_memo(memo, vs[1])
    _calc_outs(f, mnew, vs, 2, in1)
end
function _calc_outs(f, memo, vs, ind, ins...)
    ind > length(vs) && return memo, f(ins...)
    mnew, inind = output_with_memo(memo, vs[ind])
    _calc_outs(f, mnew, vs, ind+1, ins..., inind)
end

function _calc_outs2(f, memo, vs)
    ins = tuple()
    mnew = memo
    for v in vs
        mnew, newin = output_with_memo(mnew, v)
        ins = (ins..., newin)
    end
    mnew, f(ins...)
end

import ChainRulesCore: Tangent

struct Currier{F, X}
    f::F
    x::X
end
(c::Currier)(x...) = c.f(c.x, x...)
#ChainRulesCore.rrule(c::RuleConfig{>:HasReverseMode}, cf::Currier, args...) = ChainRulesCore.rrule_via_ad(c, cf.f, cf.x, args...)

function _calc_outs3(f, memo, vs)
    curriedf = f
    mnew = memo
    for v in vs
        mnew, newin = output_with_memo(mnew, v)
        curriedf = Currier(curriedf, newin)
    end
    mnew, curriedf()
end
output_with_memo(args...) = _output_with_memo(args...)
function ChainRulesCore.rrule(c::RuleConfig{>:HasReverseMode}, ::typeof(output_with_memo), memo, v)
    #@info "Call output_with_memo for $(name(v)): $(timepassed())"
    y,back = ChainRulesCore.rrule_via_ad(c, _output_with_memo, memo,  v)
    #@info "After output_with_memo for $(name(v)): $(timepassed())"
    return y, function(d)
        #@info "Call output_with_memo back for $(name(v)): $(timepassed())"
        res = back(d)
        #@info "After output_with_memo back for $(name(v)): $(timepassed())"
        res
    end
end

function ChainRulesCore.rrule(c::RuleConfig{>:HasReverseMode}, v::MutationVertex, args...) 
    #@info "      Call MutationVertex for $(name(v)): $(timepassed())"
    y, back = ChainRulesCore.rrule_via_ad(c, v.base, args...)
    #@info "      After MutationVertex for $(name(v)): $(timepassed())"
    function mutationvertex_back(d)
        #@info "      Call MutationVertex back for $(name(v)): $(timepassed())"
        res = back(d)
        #@info "      After MutationVertex back for $(name(v)): $(timepassed())"
        (Tangent{MutationVertex}(;base=res[1]), res[2:end]...)
    end
    return y, mutationvertex_back
end

function get_or_compute end
function ChainRulesCore.rrule(c::RuleConfig{>:HasReverseMode}, ::typeof(get_or_compute), f, m, v, topmemo)
    #@info "  Call get_or_compute for $(name(v)): $(timepassed())"
    y, back = ChainRulesCore.rrule_via_ad(c, _get_or_compute, f, m, v, topmemo)
    #@info "  After get_or_compute for $(name(v)): $(timepassed())"
    function get_or_compute_back(d)
        #@info "  Call get_or_compute back for $(name(v)): $(timepassed())"
        res = back(d)
        #@info "  After get_or_compute back for $(name(v)): $(timepassed())"
        res
    end
    y, get_or_compute_back
end 

calc_outs(args...) = _calc_outs(args...)
function ChainRulesCore.rrule(c::RuleConfig{>:HasReverseMode}, ::typeof(calc_outs), v, memo, vs, args...)
    #@info "    Call calc_outs for $(name(v)) with memo $memo: $(timepassed())"
    y, back = ChainRulesCore.rrule_via_ad(c, _calc_outs, v, memo, Tuple(vs), args...)
    #@info "    After calc_outs for $(name(v)) with memo $memo: $(timepassed())"
    function calc_outs_back(d)
        #@info "    Call calc_outs back for $(name(v)) with memo $memo: $(timepassed())"
        res = back(d)
        #@info "    After calc_outs back for $(name(v)) with memo $memo: $(timepassed())"
        res
    end
    y, calc_outs_back
end 

#ChainRulesCore.rrule(c::RuleConfig{>:HasReverseMode}, v::OutputsVertex, args...)= ChainRulesCore.rrule_via_ad(c, v.base, args...)
#ChainRulesCore.rrule(c::RuleConfig{>:HasReverseMode}, v::CompVertex, args...) = ChainRulesCore.rrule_via_ad(c, v.computation, args...)

function timepassed()
    t = time()
    dt = t - timestamp[]
    timestamp[] = t
    dt
end
