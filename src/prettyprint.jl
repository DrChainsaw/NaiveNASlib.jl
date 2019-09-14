
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
