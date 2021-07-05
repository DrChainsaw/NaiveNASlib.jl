# NaiveNASlib

[![Build status](https://github.com/DrChainsaw/NaiveNASlib.jl/workflows/CI/badge.svg?branch=master)](https://github.com/DrChainsaw/NaiveNASlib.jl/actions)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/NaiveNASlib.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/NaiveNASlib-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl)

NaiveNASlib is a library of functions for mutating computation graphs. It is designed with Neural Architecture Search (NAS) in mind, but can be used for any purpose where doing changes to a model architecture is desired.

It is "batteries excluded" in the sense that it is independent of both neural network implementation and search policy implementation. If you need batteries, check out [NaiveNASflux](https://github.com/DrChainsaw/NaiveNASflux.jl).

Its only contribution to this world is some help with the sometimes annoyingly complex procedure of changing an existing neural network into a new, similar yet different, neural network.

## Basic usage

```julia
]add NaiveNASlib
```

Main supported operations:
* Change the input/output size of vertices
* Parameter pruning/insertion (policy excluded)
* Add vertices to the graph
* Remove vertices from the graph
* Add edges to a vertex
* Remove edges to a vertex

For each of the above operations, NaiveNASlib makes the necessary changes to neighboring vertices to ensure that the computation graph 
is consistent w.r.t dimensions of the activations and so it to whatever extent possible represents the same function.

For complex models this can explode in complexity to the point where one might just throw in the towel completely. NaiveNASlib comes to
the rescue so that you can focus on the actual problem. Any failure to produce a valid model after mutation warrants an issue!

The price one has to pay is that the computation graph must be explicitly defined in the "language" of this library, similar to what 
some older frameworks using less modern programming languages used to do. In its defense, the main reason anyone would use this 
library to begin with is to not have to create computation graphs themselves.

Just to get started, lets create a simple graph for the summation of two numbers:
```julia
using NaiveNASlib
using Test
in1 = inputvertex("in1", 1)
in2 = inputvertex("in2", 1)

# Create a new vertex which computes the sum of in1 and in2
# Use >> to attach a name to the vertex
computation = "add" >> in1 + in2
@test computation isa MutationVertex

# CompGraph helps evaluating the whole graph as a function
graph = CompGraph([in1, in2], computation);

# Evaluate the function represented by graph
@test graph(2,3) == 5
@test graph(100,200) == 300

# The vertices function returns the vertices in topological order
@test vertices(graph) == [in1, in2, computation]
@test name.(vertices(graph)) == ["in1", "in2", "add"]
```

Now lets look at how to make use of it to modify the structure of a neural network. Since batteries are excluded, 
lets first create a tiny neural network library and do the wiring so that NaiveNASlib can work with it.

```julia
module TinyNNlib 
    using NaiveNASlib
    # A simple linear layer
    mutable struct LinearLayer{T}
        W::Matrix{T}
    end
    # Normally ones uses something like randn here, but this makes output below easier on the eyes
    LinearLayer(nin, nout) = LinearLayer(ones(Int, nout, nin))
    (l::LinearLayer)(x) = l.W * x

    # NaiveNASlib needs to know what LinearLayer considers its output size and input size
    # In this case it is the number of rows and columns of the weight matrix
    # Input size is always a vector since vertices might have multiple inputs
    NaiveNASlib.nin(l::LinearLayer) = [size(l.W, 2)]
    NaiveNASlib.nout(l::LinearLayer) = size(l.W, 1)

    # We also need to tell NaiveNASlib how to change the size of LinearLayer
    # The Δsize function will receive indices to keep from existing weights as well as where to insert new indices
    function NaiveNASlib.Δsize!(l::LinearLayer, newins::AbstractVector, newouts::AbstractVector)
        # newins is a vector of vectors as vertices may have more than one input, but LinearLayer has only one
        # The function NaiveNASlib.parselect can be used to interpret newins and newouts. 
        # We just need to tell it along which dimensions to apply them.
        l.W = NaiveNASlib.parselect(l.W, 1=>newouts, 2=>newins[])
    end

    # Helper function which creates a LinearLayer wrapped in an vertex in a computation graph.
    # This creates a Keras-like API
    linearvertex(in, outsize) = absorbvertex(LinearLayer(nout(in), outsize), in)
    export linearvertex, LinearLayer
end
```

There are a handful of other functions one can implement to e.g. provide better defaults and offer other forms of convenience, 
but here we use the bare minimum to get ourselves started. Some examples of this is provided further down.

In practice one might not want to create a whole neural network library from scratch, but rather incorporate NaiveNASlib with an existing library
in a glue package.
It might appear that this will inevitably lead to type-piracy as neither the layer definitions nor the functions (e.g. nout, nin) would 
belong to the glue package. However, it turns out that one anyways need to wrap the layers in some intermediate type. For instance, 
[NaiveNASflux](https://github.com/DrChainsaw/NaiveNASflux.jl) needs to wrap the layers from Flux in a mutable container as the layers 
themselves are not mutable.

Lets do a super simple example where we make use of the tiny neural network library to create a model and then modify it:

```julia
using .TinyNNlib

# A simple 2 layer model
invertex = inputvertex("input", 3)
layer1 = linearvertex(invertex, 4);
layer2 = linearvertex(layer1, 5);

# Vertices may be called to execute their computation alone
batchsize = 2;
batch = randn(nout(invertex), batchsize);
y1 = layer1(batch);
@test size(y1) == (nout(layer1), batchsize) == (4, 2)
y2 = layer2(y1);
@test size(y2) == (nout(layer2), batchsize) == (5, 2)

# This is just because we used nout(in) as the input size when creating the LinearLayer above
@test [nout(layer1)] == nin(layer2) == [4]

# Lets change the output size of layer1:
@test Δnout(layer1 => -2) # Returns true if successful

# And now the weight matrices have changed
@test [nout(layer1)] == nin(layer2) == [2]

# The graph is still operational :)
y1 = layer1(batch);
@test size(y1) == (nout(layer1), batchsize) == (2, 2)
y2 = layer2(y1);
@test size(y2) == (nout(layer2), batchsize) == (5 ,2)
```

As can be seen above, the consequence of changing the output size of `layer1` was that the input size of `layer2` also was changed. This is of course required for the computation graph to not throw a dimension error when being executed.

Besides the very simple graph, this mutation was trivial because the sizes of the input and output dimensions of `LinearLayer`'s parameters can change independently as they are the rows and columns of the weight matrix. This is expressed by giving the layers the mutation size trait `SizeAbsorb` in the lingo of NaiveNASlib, meaning that a change in number of input/output neurons does not propagate further in the graph.

On to the next example! As hinted above, things quickly get out of hand when using:

* Layers which require nin==nout, e.g. batch normalization and pooling.
* Element wise operations such as activation functions or just element wise arithmetics (e.g `+` used in residual connections).
* Concatenation of activations.  

Lets use a non-trivial model including all of the above. Lets first create a the model:

```julia
# First a few "normal" layers
invertex = inputvertex("input", 6);
start = linearvertex(invertex, 6);
split = linearvertex(start, nout(invertex) ÷ 3);

# When multiplying with a scalar, the output size is the same as the input size.
# This vertex type is said to be size invariant (in lack of better words).
scalarmult(v, s::Number) = invariantvertex(x -> x .* s, v)

# Concatenation means the output size is the sum of the input sizes, right?
joined = conc(scalarmult(split,2), scalarmult(split,3), scalarmult(split,5), dims=1);

# Element wise addition is of course also size invariant 
out = start + joined;

graph = CompGraph(invertex, out)
# Omitting the batch dimension happens to work with LinearLayer and we make use of it for brevity
@test graph((ones(6)))  == [78, 78, 114, 114, 186, 186]
```

Now we have a somewhat complex set of size relations at our hand since the sizes are constrained so that
  1. `start` and `joined` must have the same output size due to element wise addition.
  2. `joined` will always have 3 times the output size of `split` since there are no size absorbing vertices between them.

Modifying this graph manually would of course be manageable (albeit a bit cumbersome) if we created the model
by hand and knew it inside out. 

When things like the above emerges out of a neural architecture search things are less fun though and this is where 
use of NaiveNASlib will really pay off.

```julia
# Ok, lets try to change the size of the vertex "out".
# Before we do that, lets have a look at the sizes of the vertices in the graph to have something to compare to
@test [nout(start)] == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [6]
@test [nout(start), nout(joined)] == nin(out) == [6, 6]

# In many cases it is useful to hold on to the old graph before mutating
parentgraph = copy(graph)

# It is not possible to change the size of out by just 2
# By default, NaiveNASlib warns when this happens and then tries to make the closest possible change
# If we don't want the warning, we can tell NaiveNASlib to relax and make the closest possible change right away
@test Δnout(out => relaxed(2))

# We didn't touch the input when mutating...
@test [nout(invertex)] == nin(start) == [6]
# Start and joined must have the same size due to elementwise op.
# All three scalarmult vertices are transparent and propagate the size change to split
@test [nout(start)] == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [9]
@test [nout(start), nout(joined)] == nin(out) == [9, 9]

# parselect used by TinyNNlib will insert zeros by default when size increases. 
# This generally helps the graph maintain the same function after mutation.
# In this case we changed the size of the output layer so we don't have the exact same function though
@test graph((ones(6))) == [78, 78, 0, 114, 114, 0, 186, 186, 0]

# Copy is still intact
@test parentgraph((ones(6))) == [78, 78, 114, 114, 186, 186]

```
The core idea of NaiveNASlib is basically to annotate the type of vertex in the graph so that it knows what is the proper way to deal with the neighboring vertices when mutating a vertex.

This is done through labeling vertices into three major types:
* `SizeAbsorb`: Assumes `nout(v)` and `nin(v)` may change independently. This means that size changes are absorbed by this vertex in the sense they don't propagate further.

* `SizeStack`: Assumes `nout(v) == sum(nin(v))`. This means that size changes propagate forwards (i.e. input -> output and output -> input).

* `SizeInvariant`: Assumes `[nout(v)] == unique(nin(v))`. This means that size changes propagate both forwards and backwards as changing any input size or the output size means all others must change as well.

To use this library to mutate architectures for some neural network library basically means annotating up the above type for each layer type and connect parameter dimensions to input and output sizes (e.g. what are the input/output channel dimensions for a convolutional layer).

While we still have the complex model in scope, lets show a few more way to change the sizes. There are more examples in the built in documentation.

```julia
# Supply a utility function for telling the value of each neuron in a vertex
# NaiveNASlib will prioritize selecting the indices with higher value

# Prefer high indices:
graphhigh = copy(graph);
@test Δnout(v -> 1:nout(v), graphhigh.outputs[] => -3)
@test graphhigh((ones(6))) == [42, 0, 60, 0, 96, 0]

# Perfer low indices
graphlow = copy(graph);
@test Δnout(v -> nout(v):-1:1, graphlow.outputs[] => -3) 
@test graphlow((ones(6))) == [78, 78, 114, 114, 186, 186]

# A common approach when doing structured pruning is to prefer neurons with high magnitude.
# Here is how to set that as the default for LinearLayer.
# This is something one should probably implement in TinyNNlib instead... 
import Statistics: mean
NaiveNASlib.default_outvalue(l::LinearLayer) = mean(abs, l.W, dims=2)

graphhighmag = copy(graph);
@test Δnout(graphhighmag.outputs[] => -3) 
@test graphhighmag((ones(6))) == [78, 78, 114, 114, 186, 186]

# In many NAS applications one wants to apply random mutations to the graph
# When doing so, one might end up in situations like this:
badgraphdecinc = copy(graph);
v1, v2 = vertices(badgraphdecinc)[[3, end]]; # Imagine selecting these at random
@test Δnout(v1 => relaxed(-2))
@test Δnout(v2 => 6)
# Now we first deleted a bunch of weights, then we added new :(
@test badgraphdecinc((ones(6))) ==  [42, 0, 0, 60, 0, 0, 96, 0, 0]

# In such cases, it might be better to supply all wanted changes in one go and let 
# NaiveNASlib try to come up with a decent compromise.
goodgraphdecinc = copy(graph);
v1, v2 = vertices(goodgraphdecinc)[[3, end]];
@test Δnout(v1 => relaxed(-2), v2 => 3) # Mix relaxed and exact size changes freely
@test goodgraphdecinc((ones(6))) == [78, 78, 0, 0, 114, 114, 0, 0, 186, 186, 0, 0] 

# It is also possible to change the input direction, but it requires specifying a size change for each input
graphΔnin = copy(graph);
v1, v2 = vertices(graphΔnin)[end-1:end];
@test Δnin(v1 => (3, relaxed(2), missing), v2 => relaxed((1,2))) # Use missing to signal "don't care"
@test nin(v1) == [6, 6, 6] # Sizes are tied to nout of split so they all have to be equal
@test nin(v2) == [18, 18] # Sizes are tied due to elementwise addition

# Another popular pruning strategy is to just remove the x% of params with lowest value
# This can be done by just not putting any size requirements and assign negative value
graphprune40 = copy(graph);
Δsize!(graphprune40) do v
    # Assign no value to SizeTransparent vertices
    trait(v) isa NaiveNASlib.SizeTransparent && return 0
    value = NaiveNASlib.default_outvalue(v)
    return value .- 0.4mean(value)
end
@test nout.(vertices(graphprune40)) == [6, 6, 2, 2, 2, 2, 6, 6]
#Compare to original:
@test nout.(vertices(graph))        == [6, 9, 3, 3, 3, 3, 9, 9]
```

Here is a closer look at how the weight matrices are changed in case this was not clear from the graph outputs above:

```julia
# Return layer just so we can easily look at it
function vertexandlayer(in, outsize)
    nparam = nout(in) * outsize
    l = LinearLayer(collect(reshape(1:nparam, :, nout(in))))
    return absorbvertex(l, in), l
end

# Make a simple model
invertices = inputvertex.(["in1", "in2"], [3,4])
v1, l1 = vertexandlayer(invertices[1], 4)
v2, l2 = vertexandlayer(invertices[2], 3)
merged = conc(v1, v2, dims=1)
v3, l3 = vertexandlayer(merged, 2)
graph = CompGraph(invertices, v3)

# These weights are probably not useful in a real neural network.
# They are just to make it easier to spot what has changed after size change below.
@test l1.W ==
[ 1 5  9 ; 
  2 6 10 ; 
  3 7 11 ;
  4 8 12 ]

@test l2.W ==
[ 1 4 7 10 ;
  2 5 8 11 ; 
  3 6 9 12 ]

@test l3.W ==
[ 1  3  5  7   9  11  13 ;
  2  4  6  8  10  12  14 ]

# Now, lets decrease v1 by 1 and force merged to retain its size 
# which in turn forces v2 to grow by 1
# Give high value to neurons 1 and 3 of v2, same for all others...
@test Δnout(v2 => -1, merged => 0) do v
    v == v2 ? [10, 1, 10] : ones(nout(v))
end

# v1 got a new row of parameters at the end
@test l1.W ==
[ 1  5   9 ;
  2  6  10 ;
  3  7  11 ;
  4  8  12 ;
  0  0   0 ]

# v2 chose to drop its middle row as it was the output neuron with lowest value
@test l2.W ==
[ 1 4 7 10 ;
  3 6 9 12 ]

# v3 dropped the second to last column (which is aligned to the middle row of v2)
# and got new parameters in column 5 (which is aligned to the last row of v1)
@test l3.W ==
[  1  3  5  7  0   9  13 ;
   2  4  6  8  0  10  14 ]
```


Lets just do a few quick examples of the other use cases.

Add a vertex to a graph:
```julia
invertex = inputvertex("input", 3)
layer1 = linearvertex(invertex, 5)
graph = CompGraph(invertex, layer1)

# nv(g) is shortcut for length(vertices(g))
@test nv(graph) == 2
@test graph(ones(3)) == [3,3,3,3,3]

# Insert a layer between invertex and layer1
@test insert!(invertex, vertex -> linearvertex(vertex, nout(vertex))) # True if success

@test nv(graph) == 3
@test graph(ones(3)) == [9, 9, 9, 9, 9]
```

Remove a vertex from a graph:
```julia
invertex = inputvertex("input", 3)
layer1 = linearvertex(invertex, 5)
layer2 = linearvertex(layer1, 4)
graph = CompGraph(invertex, layer2)

@test nv(graph) == 3
@test graph(ones(3)) == [15, 15, 15, 15]

# Remove layer1 and change nin of layer2 from 5 to 3
# Would perhaps have been better to increase nout of invertex, but it is immutable
@test remove!(layer1) # True if success

@test nv(graph) == 2
@test graph(ones(3)) == [3, 3, 3, 3]
```

Add an input edge to a vertex:
```julia
invertices = inputvertex.(["input1", "input2"], [3, 2])
layer1 = linearvertex(invertices[1], 4)
layer2 = linearvertex(invertices[2], 4)
add = layer1 + layer2
out = linearvertex(add, 5)
graph = CompGraph(invertices, out)

@test nin(add) == [4, 4]
# Two inputs this time, remember?
@test graph(ones(3), ones(2)) == [20, 20, 20, 20, 20]

# This graph is not interesting enough for there to be a good showcase for adding a new edge.
# Lets create a new layer which has a different output size just to see how things change
# The only vertex which support more than one input is add
layer3 = linearvertex(invertices[2], 6)
@test create_edge!(layer3, add) # True if success

# By default, NaiveNASlib will try to increase the size in case of a mismatch
@test nin(add) == [6, 6, 6]
@test graph(ones(3), ones(2)) == [28, 28, 28, 28, 28] 
```

Remove an edge from a vertex:
```julia
invertex = inputvertex("input", 4)
layer1 = linearvertex(invertex, 3)
layer2 = linearvertex(invertex, 5)
merged = conc(layer1, layer2, layer1, dims=1)
out = linearvertex(merged, 3)
graph = CompGraph(invertex, out)

@test nin(merged) == [3, 5, 3]
@test graph(ones(4)) == [44, 44, 44]

@test remove_edge!(layer1, merged) # True if success

@test nin(merged) == [5, 3]
@test graph(ones(4)) == [32, 32, 32]
```

## Advanced usage

The previous examples have been focused on giving an overview of the purpose of this library. For more advanced usage, there are many of ways to customize the behavior and in other ways alter or hook in to the functionality. Here are a few of the most important.

### Strategies

For more or less all operations which mutate the graph, it is possible achieve fine grained control of the operation through selecting a strategy.

Here is an example of strategies for changing the size:

```julia
# A simple graph where one vertex has a constraint for changing the size.
invertex = inputvertex("in", 3)
layer1 = linearvertex(invertex, 4)
# joined can only change in steps of 2
joined = conc(scalarmult(layer1, 2), scalarmult(layer1, 3), dims=1)          

# all_in_graph finds all vertices in the same graph as the given vertex
verts = all_in_graph(joined)

# Strategy to try to change it by one and throw an error when not successful
exact_or_fail = ΔNoutExact(joined => 1; fallback=ThrowΔSizeFailError("Size change failed!!"))

# Note that we now call Δsize instead of Δnout as the direction is given by the strategy
@test_throws NaiveNASlib.ΔSizeFailError Δsize!(exact_or_fail, verts)

# No change was made
@test nout(joined) == 2*nout(layer1) == 8

# Try to change by one and fail silently when not successful
exact_or_noop = ΔNoutExact(joined=>1;fallback=ΔSizeFailNoOp())

@test !Δsize!(exact_or_noop, verts) 

# No change was made
@test nout(joined) == 2*nout(layer1) == 8

# In many cases it is ok to not get the exact change which was requested
relaxed_or_fail = ΔNoutRelaxed(joined=>1;fallback=ThrowΔSizeFailError("This should not happen!!"))

@test Δsize!(relaxed_or_fail, verts)

# Changed by two as this was the smallest possible change
@test nout(joined) == 2*nout(layer1) == 10

# Logging when fallback is applied is also possible
using Logging
# Yeah, this is not easy on the eyes, but it gets the job done...
exact_or_log_then_relax = ΔNoutExact(joined=>1; fallback=LogΔSizeExec("Exact failed, relaxing", Logging.Info, relaxed_or_fail))

@test_logs (:info, "Exact failed, relaxing") Δsize!(exact_or_log_then_relax, verts)

@test nout(joined) == 2*nout(layer1) == 12
```
A similar pattern is used for most other mutating operations. Use the built-in documentation to explore the options until I find the energy and time to write proper documentation. As I could not let go of the OO habit of having abstract base types for everything, the existing strategies can be discovered using `subtypes` as a stop-gap solution.

### Traits

A variant (bastardization?) of the [holy trait](https://docs.julialang.org/en/v1/manual/methods/#Trait-based-dispatch-1) pattern is used to annotate the type of a vertex. In the examples above the three 'core' types `SizeAbsorb`, `SizeStack` and `SizeInvariant` are shown, but it is also possible to attach other information and behaviors by freeriding on this mechanism.

This is done by adding the argument `traitdecoration` when creating a vertex and supplying a function which takes a trait and return a new trait (which typically wraps the input).

Some examples:

```julia
noname = linearvertex(inputvertex("in", 2), 2)
@test name(noname) == "MutationVertex::SizeAbsorb"

# Naming vertices is so useful for logging and debugging I almost made it mandatory
named = absorbvertex(LinearLayer(2, 3), inputvertex("in", 2), traitdecoration = t -> NamedTrait(t, "named layer"))
@test name(named) == "named layer"

# Speaking of logging...
layer1 = absorbvertex(LinearLayer(2, 3), inputvertex("in", 2), traitdecoration = t -> SizeChangeLogger(NamedTrait(t, "layer1")))

# What info is shown can be controlled by supplying an extra argument to SizeChangeLogger
nameonly = NameInfoStr()
layer2 = absorbvertex(LinearLayer(nout(layer1), 4), layer1, traitdecoration = t -> SizeChangeLogger(nameonly, NamedTrait(t, "layer2")))

@test_logs(
(:info, "Change nout of layer1, inputs=[in], outputs=[layer2], nin=[2], nout=[4], SizeAbsorb() by [1, 2, 3, -1]"),
(:info, "Change nin of layer2 by [1, 2, 3, -1]"), # Note: less verbose compared to layer1 due to NameInfoStr
Δnout(layer1, 1))

# traitdecoration works exactly the same for conc and invariantvertex as well, no need for an example

# For more elaborate traits with element wise operations one can use traitconf and >>
add = traitconf(t -> SizeChangeLogger(NamedTrait(t, "layer1 + layer2"))) >> layer1 + layer2
@test name(add) == "layer1 + layer2"

@test_logs(
(:info, "Change nout of layer1, inputs=[in], outputs=[layer2, layer1 + layer2], nin=[2], nout=[5], SizeAbsorb() by [1, 2, 3, 4, -1]"),
(:info, "Change nin of layer2 by [1, 2, 3, 4, -1]"),
(:info, "Change nout of layer2 by [1, 2, 3, 4, -1]"),
(:info, "Change nin of layer1 + layer2, inputs=[layer1, layer2], outputs=[], nin=[5, 5], nout=[5], SizeInvariant() by [1, 2, 3, 4, -1] and [1, 2, 3, 4, -1]"),
(:info, "Change nout of layer1 + layer2, inputs=[layer1, layer2], outputs=[], nin=[5, 5], nout=[5], SizeInvariant() by [1, 2, 3, 4, -1]"),
Δnout(layer1, 1))

# When creating own trait wrappers, remember to subtype DecoratingTrait or else there will be pain!

# Wrong!! Not a subtype of DecoratingTrait
struct PainfulTrait{T<:MutationTrait} <: MutationTrait
    base::T
end
painlayer = absorbvertex(LinearLayer(2, 3), inputvertex("in", 2), traitdecoration = PainfulTrait)

# Now one must implement a lot of methods for PainfulTrait...
@test_throws MethodError Δnout(painlayer, 1)

# Right! Is a subtype of DecoratingTrait
struct SmoothSailingTrait{T<:MutationTrait} <: DecoratingTrait
    base::T
end
# Just implement base and all will be fine
NaiveNASlib.base(t::SmoothSailingTrait) = t.base

smoothlayer = absorbvertex(LinearLayer(2, 3), inputvertex("in", 2), traitdecoration = SmoothSailingTrait)

@test Δnout(smoothlayer, 1)
@test nout(smoothlayer) == 4

```

### Graph instrumentation and modification

In many cases it is desirable to change things like traits of an existing graph. This can be achieved by supplying an extra argument when copying the graph. The extra argument is a function which determines how each individual component of the graph shall be copied.

Depending on what one wants to achieve, it can be more or less messy. Here is a pretty messy example:

```julia
invertex = inputvertex("in", 2)
layer1 = linearvertex(invertex, 3)
layer2 = linearvertex(layer1, 4)

graph = CompGraph(invertex, layer2)

@test name.(vertices(graph)) == ["in", "MutationVertex::SizeAbsorb", "MutationVertex::SizeAbsorb"]

# Ok, lets add names to layer1 and layer2 and change the name of invertex

# Lets first define the default: Fallback to "clone"
# clone is the existing function to copy things in this manner as I did not want to override Base.copy
copyfun(args...;cf) = clone(args...;cf=cf) # Keyword argument cf is the function to use for copying all fields of the input

# Add a name to layer1 and layer2
function copyfun(v::MutationVertex,args...;cf)
    # This is probably not practical to do in a real graph, so make sure you have names when first creating it...
    name = v == layer1 ? "layer1" : "layer2"
    addname(args...;cf) = clone(args...;cf=cf)
    addname(t::SizeAbsorb;cf) = NamedTrait(t, name) # SizeAbsorb has no fields, otherwise we would have had to run cf for each one of them...
    clone(v, args...;cf=addname)
end

# Change name of invertex
# Here we can assume that invertex name is unique in the whole graph or else we would have had to use the above way
copyfun(s::String; cf) = s == name(invertex) ? "in changed" : s

# Now supply copyfun when copying the graph.
# I must admit that thinking about what this does makes me a bit dizzy...
namedgraph = copy(graph, copyfun)

@test name.(vertices(namedgraph)) == ["in changed", "layer1", "layer2"]
```

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
