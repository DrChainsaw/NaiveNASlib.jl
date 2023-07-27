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

function get_or_compute(f, m::LinkedMemo, key, topmemo)
    memokey(m) === key && return (topmemo, memovalue(m))
    get_or_compute(f, m.next, key, topmemo)
end
function get_or_compute(f, m::Memo, key, topmemo)
    memokey(m) === key && return (topmemo, memovalue(m))
    f(topmemo, key)
end

# Non-recursive variant of the above
function get_or_compute2(f, m::AbstractMemo, key, topmemo)
    while !isa(m, Memo)
        memokey(m) === key && return topmemo, memovalue(m)
        m = m.next
    end
    memokey(m) === key && return topmemo, memovalue(m)
    f(topmemo, key)
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
    f(a, key)
end
_memoize(a::AbstractArray, k, v) = vcat(a, k => v)

compute_graph(memo::AbstractMemo, v) = last(output_with_memo(memo, v))
compute_graph(memo::AbstractArray, v) = last(output_with_memo(memo, v))

output_with_memo(memo, v::AbstractVertex) = get_or_compute(memo, v) do mmemo, vv
    mnew, ins = _calc_outs3(mmemo, inputs(vv))
    out = vv(ins...)
    _maybe_memoize(mnew, vv, out), out
end
# This seems to be worse for compile times compared to just having an if statement in output_with_memo
_maybe_memoize(memo, v, out) = _memoize(memo, v, out) # For vertices which don't support outputs
_maybe_memoize(memo, v::MutationVertex, out) =  length(outputs(v)) > 1 ? _memoize(memo, v, out) : memo

# ChainRules/Zygote friendly combination of map and reduction (map over vs, reduction of memo)
# Tuple might seem unnecessary/detrimental, but without it compile times randomly blow up
_calc_outs(memo, vs::AbstractArray) = _calc_outs(memo, Tuple(vs))
function _calc_outs(memo, vs::Tuple)
    mnew, in1 = output_with_memo(memo, vs[1])
    _calc_outs(mnew, vs, 2, in1)
end
function _calc_outs(memo, vs::Tuple, ind, ins...)
    ind > length(vs) && return memo, ins
    mnew, inind = output_with_memo(memo, vs[ind])
    _calc_outs(mnew, vs, ind+1, ins..., inind)
end

_calc_outs2(memo, vs::AbstractArray) = _calc_outs2(memo, Tuple(vs))
function _calc_outs2(memo, vs::Tuple)
    ins = tuple()
    mnew = memo
    for v in vs
        mnew, newin = output_with_memo(mnew, v)
        ins = (ins..., newin)
    end
    mnew, ins
end

import Base.Cartesian: @nexprs

_calc_outs3(memo, vs::AbstractArray) = _calc_outs3(memo, Tuple(vs))
@generated function _calc_outs3(memo, vs::Tuple{Vararg{Any, N}}) where N
    outs = ntuple( i -> Symbol(:out_, i), Val(N))
    quote
        mnew = memo
        @nexprs $N j->((mnew, out_j) = output_with_memo(mnew, vs[j]))
        mnew, tuple($(outs...))
    end
end


