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
    memokey(m) === key && return topmemo, memovalue(m)
    get_or_compute(f, m.next, key, topmemo)
end
function get_or_compute(f, m::Memo, key, topmemo)
    memokey(m) === key && return topmemo, memovalue(m)
    f(key)
end

Base.pairs(m::Memo) = tuple(memokey(m) => memovalue(m))
Base.pairs(m::LinkedMemo) = Iterators.flatten((pairs(m.this), pairs(m.next)))

function Base.show(io::IO, m::AbstractMemo) 
    print(io, "Memo(")
    namearr = map(pairs(m)) do (k, v)
        k isa AbstractVertex && return name(k) => v
        k => v 
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


compute_graph(memo::AbstractMemo, v) = last(output_with_memo(memo, v))
compute_graph(memo::AbstractArray, v) = last(output_with_memo(memo, v))
function output_with_memo(memo, v::AbstractVertex)
    get_or_compute(memo, v) do vv
        mnew, inpt = _calc_outs(memo, inputs(vv))
        out = vv(inpt...)
        length(outputs(vv)) > 1 ? (_memoize(mnew, vv, out), out) : (mnew, out)
    end 
end

# ChainRules/Zygote friendly combination of map and reduction (map over vs, reduction of memo)
function _calc_outs(memo, vs)
    mnew, out = output_with_memo(memo, vs[1])
    _calc_outs(tuple(out), mnew, vs, 2)
end
function _calc_outs(outs, memo, vs, ind)
    ind > length(vs) && return memo, outs
    mnew, out = output_with_memo(memo, vs[ind])
    _calc_outs((outs..., out), mnew, vs, ind+1)
end
