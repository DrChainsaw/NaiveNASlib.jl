abstract type AbstractMemo end

struct Memo{VT, OT} <: AbstractMemo
    key::VT
    value::OT
end
Memo(p::Pair) = Memo(first(p), last(p))
Memo(itr) = foldl((m, e) -> _memoize(m, e...), Iterators.drop(itr, 1); init=Memo(first(itr)...))


init_memo(v::AbstractVertex, x) = Memo(v, x)
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

get_or_compute(f, m::AbstractMemo, key) = get_or_compute3(f, m, key, m)

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

# Generated variant of the above
function get_or_compute_expr(f, m::Type{<:LinkedMemo{PT}}, key, topmemo) where PT
    ex = quote
        memokey(m) === key && return topmemo, memovalue(m)
        m = m.next
    end
    append!(ex.args, get_or_compute_expr(f, PT, key, topmemo).args)
    return ex
end
function get_or_compute_expr(f, m::Type{<:Memo}, key, topmemo)
    quote
        memokey(m) === key && return topmemo, memovalue(m)
        f(topmemo, key)
    end
end


@generated function get_or_compute3(f, m::AbstractMemo, key, topmemo)
    get_or_compute_expr(f, m, key, topmemo)
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

compute_graph(memo, v::AbstractVertex) = last(output_with_memo(memo, v))
compute_graph(memo, vs::Tuple) = last(_calc_outs(memo, vs))

output_with_memo(memo, v::AbstractVertex) = get_or_compute(memo, v) do mmemo, vv
    mnew, ins = _calc_outs(mmemo, inputs(vv))
    out = vv(ins...)
    (vv isa MutationVertex && length(outputs(vv)) > 1) ? (_memoize(mnew, vv, out), out) : (mnew, out)
end

function _calc_outs_expr(memoname, vsname, ::Type{<:Tuple{Vararg{Any, N}}}) where N
    outs = ntuple( i -> Symbol(:out_, i), Val(N))
    calcexpr = map(i -> :((mnew, $(outs[i])) = output_with_memo(mnew, $vsname[$i])), 1:N)
    quote
        mnew = $memoname
        $(calcexpr...)
        mnew, tuple($(outs...))
    end
end


_calc_outs(memo, vs::AbstractArray) = _calc_outs(memo, Tuple(vs))
@generated function _calc_outs(memo, vs::Tuple{Vararg{Any, N}}) where N
    _calc_outs_expr(:memo, :vs, vs)
end

