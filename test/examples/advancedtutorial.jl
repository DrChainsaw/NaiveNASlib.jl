md"""
# Advanced Tutorial

The previous examples have been focused on giving an overview of the purpose of this library using the simple high level API. For more advanced usage, there are
many of ways to customize the behavior and in other ways alter or hook in to the functionality. Some of the more important concepts are described below.

To make it more convenient to extend `NaiveNASlib`, the two submodules `NaiveNASlib.Advanced` and `NaiveNASlib.Extend` export most of the useful stuff, such as 
abstract types and composable strategies. For now they are also part of the public API, but in future releases they might be moved to separate subpackages so 
that they can be versioned separately (e.g NaiveNASlibCore).
"""
using NaiveNASlib.Advanced, NaiveNASlib.Extend

md"""
## Strategies

For more or less all operations which mutate the graph, it is possible achieve fine grained control of the operation through selecting a strategy.

Here is an example of strategies for changing the size:
"""

@testset "Strategies" begin #hide
# A simple graph where one vertex has a constraint for changing the size.
invertex = inputvertex("in", 3)
layer1 = linearvertex(invertex, 4)
# `joined` can only change in steps of 2.
joined = conc(scalarmult(layer1, 2), scalarmult(layer1, 3), dims=1)          

# Strategy to try to change it by one and throw an error when not successful.
exact_or_fail = ΔNoutExact(joined => 1; fallback=ThrowΔSizeFailError("Size change failed!!"))

# Note that we now call `Δsize!` instead of `Δnout!` as the wanted action is given by the strategy.
@test_throws NaiveNASlib.ΔSizeFailError Δsize!(exact_or_fail)
@test nout(joined) == 2*nout(layer1) == 8 # No change was made.

# Try to change by one and fail silently when not successful.
exact_or_noop = ΔNoutExact(joined=>1;fallback=ΔSizeFailNoOp())

@test !Δsize!(exact_or_noop) 
@test nout(joined) == 2*nout(layer1) == 8 # No change was made.

# In many cases it is ok to not get the exact change which was requested.
relaxed_or_fail = ΔNoutRelaxed(joined=>1;fallback=ThrowΔSizeFailError("This should not happen!!"))

@test Δsize!(relaxed_or_fail)
## Changed by two as this was the smallest possible change.
@test nout(joined) == 2*nout(layer1) == 10

# Logging when fallback is applied is also possible.
using Logging: Info
exact_or_log_then_relax = ΔNoutExact(joined=>1; 
                                        fallback=LogΔSizeExec(
                                                        "Exact failed, relaxing", 
                                                        Info, 
                                                        relaxed_or_fail))

@test_logs (:info, "Exact failed, relaxing") Δsize!(exact_or_log_then_relax)
@test nout(joined) == 2*nout(layer1) == 12

# If one wants to see every size change we can set up an `AfterΔSizeCallback` strategy to log it for us like this:
exact_or_log_then_relax_verbose = logafterΔsize(v -> "some vertex";base=exact_or_log_then_relax)

@test_logs( 
    (:info, "Exact failed, relaxing"),
    (:info, r"Change nin of some vertex"),
    (:info, r"Change nout of some vertex"),
    match_mode=:any,
    Δsize!(exact_or_log_then_relax_verbose))    
end #hide

md"""
A similar pattern is used for most other mutating operations. Use the built-in documentation to explore the options until I
find the energy and time to write proper documentation. As I could not let go of the OO habit of having abstract base types for 
everything, the existing strategies can be discovered using `subtypes` as a stop-gap solution.
"""
md"""
## Traits

A variant (bastardization?) of the [holy trait](https://docs.julialang.org/en/v1/manual/methods/#Trait-based-dispatch-1) pattern is used to 
annotate the type of a vertex. In the examples above the three 'core' types `SizeAbsorb`, `SizeStack` and `SizeInvariant` are shown, but it is 
also possible to attach other information and behaviors by freeriding on this mechanism.

This is done by adding the argument `traitdecoration` when creating a vertex and supplying a function which takes a trait and return a new trait 
(which typically wraps the input).

"""

@testset "Traits" begin #hide
# Naming vertices is so useful for logging and debugging I almost made it mandatory. 
#
# If a vertex does not have the named trait, `name` will return a generic string. Compare
noname = linearvertex(inputvertex("in", 2), 2)
@test name(noname) == "MutationVertex::SizeAbsorb"
# with
hasname = absorbvertex(LinearLayer(2, 3), inputvertex("in", 2), traitdecoration = t -> NamedTrait("named layer", t))
@test name(hasname) == "named layer"

# `AfterΔSizeTrait` can be used to attach an `AbstractAfterΔSizeStrategy` to an individual vertex.
# In this case we use `logafterΔsize` from the example above.
verbose_vertex_info(v) = string(name(v),
                                 " with inputs=[", join(name.(inputs(v)),  ", "), 
                                "] and outputs=[", join(name.(outputs(v)), ", "), ']')
named_verbose_logging(t) = AfterΔSizeTrait(
                                        logafterΔsize(verbose_vertex_info),
                                        NamedTrait("layer1", t))
layer1 = absorbvertex(  LinearLayer(2, 3), 
                        inputvertex("in", 2), 
                        traitdecoration = named_verbose_logging)
# The above is a mouthful, but `NaiveNASlib.Advanced` exports the `named` and `logged` functions for convenience
layer2 = absorbvertex(  LinearLayer(nout(layer1), 4), 
                        layer1; 
                        traitdecoration = logged(name) ∘ named("layer2"))

# Now logs for `layer2` are less verbose than logs for `layer1` due to `name` being used to print the
# vertex instead of `verbose_vertex_info`.
@test_logs(
(:info, "Change nout of layer1 with inputs=[in] and outputs=[layer2] by [1, 2, 3, -1]"),
(:info, "Change nin of layer2 by [1, 2, 3, -1]"), 
Δnout!(layer1, 1))

# For more elaborate traits with elementwise operations one can use traitconf and `>>`
add = traitconf(logged(verbose_vertex_info) ∘ named("layer1+layer2")) >> layer1 + layer2
@test name(add) == "layer1+layer2"

@test_logs(
(:info, "Change nout of layer1 with inputs=[in] and outputs=[layer2, layer1+layer2] by [1, 2, 3, 4, -1]"),
(:info, "Change nin of layer2 by [1, 2, 3, 4, -1]"),
(:info, "Change nout of layer2 by [1, 2, 3, 4, -1]"),
(:info, "Change nin of layer1+layer2 with inputs=[layer1, layer2] and outputs=[] by [1, 2, 3, 4, -1] and [1, 2, 3, 4, -1]"),
(:info, "Change nout of layer1+layer2 with inputs=[layer1, layer2] and outputs=[] by [1, 2, 3, 4, -1]"),
Δnout!(layer1, 1))

# When creating own trait wrappers, remember to subtype `DecoratingTrait` or else there will be pain!
#
# Wrong!! Not a subtype of `DecoratingTrait`
struct PainfulTrait{T<:MutationTrait} <: MutationTrait
    base::T
end
painlayer = absorbvertex(   LinearLayer(2, 3), 
                            inputvertex("in", 2);
                            traitdecoration = PainfulTrait)

# Now one must implement a lot of methods for `PainfulTrait`...
@test_throws MethodError Δnout!(painlayer, 1)

# Right!! Is a subtype of `DecoratingTrait`.
struct SmoothSailingTrait{T<:MutationTrait} <: DecoratingTrait
    base::T
end
# Just implement `base` and all will be fine.
NaiveNASlib.base(t::SmoothSailingTrait) = t.base

smoothlayer = absorbvertex( LinearLayer(2, 3), 
                            inputvertex("in", 2);
                            traitdecoration = SmoothSailingTrait)

@test Δnout!(smoothlayer, 1)
@test nout(smoothlayer) == 4
end #hide

md"""
## Graph instrumentation and modification

In many cases it is desirable to change things like traits of an existing graph. This can be achieved through [Functors.jl](https://github.com/FluxML/Functors.jl), often through clever usage of the `walk` function.

Depending on what one wants to achieve, it can be more or less messy. Here is a pretty messy example:
"""

@testset "Graph instrumentation and modification" begin #hide
invertex = inputvertex("in", 2)
layer1 = linearvertex(invertex, 3)
layer2 = linearvertex(layer1, 4)

graph = CompGraph(invertex, layer2)

@test name.(vertices(graph)) == ["in", "MutationVertex::SizeAbsorb", "MutationVertex::SizeAbsorb"]

# Ok, lets add names to `layer1` and `layer2` and change the name of `invertex`

# Add a name to `layer1` and `layer2`
function walk(f, v::MutationVertex)
    ## This is probably not practical to do in a real graph, so make sure you have names when first creating it...
    name = v == layer1 ? "layer1" : "layer2"
    addname(x) = x
    ## SizeAbsorb has no fields, otherwise we would have had to use walk to wrap it
    addname(t::SizeAbsorb) = NamedTrait(name, t)
    Functors._default_walk(v) do x
        fmap(addname, x; walk)
    end
end

# Change name of `invertex` once we get there.
# We could also just have made a string version of `addname` above 
# since there are no other Strings in the graph, but this is safer.
function walk(f, v::InputVertex)
    rename(x) = x
    rename(s::String) = "in changed"
    Functors._default_walk(v) do x
        fmap(rename, x; walk)
    end
end

# Everything else just gets functored as normal.
walk(f, x) = Functors._default_walk(f, x) 

# I must admit that thinking about what this does makes me a bit dizzy...
namedgraph = fmap(identity, graph; walk)

@test name.(vertices(namedgraph)) == ["in changed", "layer1", "layer2"]
end #hide

