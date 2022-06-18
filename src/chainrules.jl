
const enable_explicit_gradients = Ref(false)

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(output!), memo, v)
    rrule_via_ad(config, output_rrule!, memo, v)
end

# Workaround for https://github.com/FluxML/Zygote.jl/issues/1111
# and https://github.com/FluxML/Zygote.jl/issues/1243
# Only purpose is to return NoTangent, so whole function can be deleted
# if/when issues are resolved. Done forget to delete enable_explicit_gradients too then!
output_rrule!(args...) = _output_rrule!(args...)
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(output_rrule!), memo, v)
    res, back = rrule_via_ad(config, _output_rrule!, memo, v)
    return res, function (d)
        bres = back(d)
        return enable_explicit_gradients[] ? bres : (NoTangent(), NoTangent(), NoTangent())
    end
end

function _output_rrule!(memo, v::AbstractVertex)
    # rrule for get! not implemented, so we need to check the dict twice
    v in keys(memo) && return memo[v]
    inpt = map(iv -> output_rrule!(memo, iv),  inputs(v))
    memo[v] = v(inpt...)
end

