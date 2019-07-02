# NaiveNASlib

[![Build Status](https://travis-ci.com/DrChainsaw/NaiveNASlib.jl.svg?branch=master)](https://travis-ci.com/DrChainsaw/NaiveNASlib.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/NaiveNASlib.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/NaiveNASlib-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl)

Work in progress! Limited functionality (and bugs, but there will always be bugs).

NaiveNASlib is a library of functions for mutating computation graphs in order to support Neural Architecture Search (NAS).

It is "batteries excluded" in the sense that it is independent of both neural network implementation and search policy implementation.

Its only contribution to this world is some help with the sometimes annoyingly complex procedure of changing an existing neural network architecture into a new, similar yet different, neural network architecture.

## Basic usage

```julia
Pkg.add("https://github.com/DrChainsaw/NaiveNASlib.jl")
```

The price one has to pay is that the computation graph must be explicitly defined in the "language" of this library, similar to what some older frameworks using less modern programming languages used to do. In its defense, the sole reason anyone would use this library to begin with is to not have to create computation graphs themselves.

Main supported use cases:
* Change the input/output size of vertices
* Parameter pruning (policy excluded)
* Remove a vertex from the graph
* Add a vertex to the graph
* Add/remove edges to/from vertex

For each of the above operations, NaiveNASlib makes the necessary changes to neighbouring vertices to ensure that the computation graph is consistent w.r.t dimensions of the activations.

Just to get started, lets create a simple graph for the summation of two numbers:
```julia
in1, in2 = InputVertex.(("in1", "in2"));

computation = CompVertex(+, in1, in2);

graph = CompGraph([in1, in2], computation);

using Test

@test graph(2,3) == 5
```

Now for a more to the point example. The vertex types used above does not contain any information needed to mutate the graph. This might be a sign of OOP damage, but in order to do so, we need to wrap them in a vertex type which supports mutation.

```julia
# First we need something to mutate. Batteries excluded, remember?
mutable struct SimpleLayer
    W
    SimpleLayer(W) = new(W)
    SimpleLayer(nin, nout) = new(ones(Int, nin,nout))
end
(l::SimpleLayer)(x) = x * l.W

# Helper function which creates a mutable layer.
layer(in, outsize) = absorbvertex(SimpleLayer(nout(in), outsize), outsize, in, mutation=IoSize)

input = inputvertex("input", 3)
layer1 = layer(input, 4);
layer2 = layer(layer1, 5);

@test [nout(layer1)] == nin(layer2) == [4]

#Lets change the output size of layer1:
Δnout(layer1, -2);

@test [nout(layer1)] == nin(layer2) == [2]

#Lets change the output size of layer1:
Δnout(layer1, -2);

@test [nout(layer1)] == nin(layer2) == [2]
```
As can be seen above, the consequence of changing the output size of `layer1` was that the input size of `layer2` also was changed.

This mutation was trivial because both layers are of the type `SizeAbsorb`, meaning that a change in number of inputs/outputs does not propagate further in the graph.

Lets do a non-trivial example:
```julia

scalarmult(v, f::Integer) = vertex(x -> x .* f, nout(v), SizeInvariant(), v)

input = inputvertex("input", 6);
start = layer(input, 6);
split = layer(start, div(nout(input) , 3));
joined = conc(scalarmult(split, 2), scalarmult(split,3), scalarmult(split,5), dims=2);
out = start + joined;

@test [nout(input)] == nin(start) == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [6]
@test [nout(start), nout(joined)] == nin(out) == [6, 6]

graph = CompGraph(input, out)
@test graph((ones(Int, 1,6))) == [78  78  114  114  186  186]

# Ok, lets try to reduce the size of the vertex "out".
# First we need to realize that we can only change it by integer multiples of 3
# This is because it is connected to "split" through three paths which require nin==nout

# We need this information from the layer. Some layers have other requirements
NaiveNASlib.minΔnoutfactor(::SimpleLayer) = 1
NaiveNASlib.minΔninfactor(::SimpleLayer) = 1

@test minΔnoutfactor(out) == minΔninfactor(out) == 3

# Next, we need to define how to mutate our SimpleLayer
NaiveNASlib.mutate_inputs(l::SimpleLayer, newInSize) = l.W = ones(Int, newInSize, size(l.W,2))
NaiveNASlib.mutate_outputs(l::SimpleLayer, newOutSize) = l.W = ones(Int, size(l.W,1), newOutSize)

#In some cases it is useful to hold on to the old graph before mutating
# To do so, we need to define the clone operation for our SimpleLayer
NaiveNASlib.clone(l::SimpleLayer) = SimpleLayer(l.W)
parentgraph = copy(graph)

Δnin(out, 3)

# We didn't touch the input when mutating...
@test [nout(input)] == nin(start) == [6]
# Start and joined must have the same size due to elementwise op.
# All three scalarmult vertices are transparent and propagate the size change to split
@test [nout(start)] == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [9]
@test [nout(start), nout(joined)] == nin(out) == [9, 9]

# However, this only updated the mutation metadata, not the actual layer.
# There are some slightly annoying and perhaps overthought reasons to this
# I will document them once things crystalize a bit more
@test graph((ones(Int, 1,6))) == [78  78  114  114  186  186]

# To mutate the graph, we need to apply the mutation:
apply_mutation(graph);

@test graph((ones(Int, 1,6))) == [114  114  114  168  168  168  276  276  276]

# Copy is still intact
@test parentgraph((ones(Int, 1,6))) == [78  78  114  114  186  186]

```

As seen above, things get a little bit out of hand when using:

* Layers which require nin==nout such as batch normalization
* Element wise operations
* Concatenation of activations  

The core idea of NaiveNASlib is basically to annotate the type of vertex in the graph so that functions know what is the proper way to deal with the neighbouring vertices when mutating a vertex.

This is done through labeling vertices into three major types:
* `SizeAbsorb`: Assumes `nout(v)` and `nin(v)` may change independently. This means that size changes are absorbed by this vertex in the sense they don't propagate further.

* `SizeStack`: Assumes `nout(v) == sum(nin(v))`. This means that size changes propagate forwards (i.e input -> input and output -> output).

* `SizeInvariant`: Assumes `[nout(v)] == unique(nin(v))`. This means that size changes propagate both forwards and backwards as changing any input size or the output size means all others must change as well.

To use this library to mutate architectures for some neural network library basically means annotating up the above type for each layer in the neural network library.  

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
