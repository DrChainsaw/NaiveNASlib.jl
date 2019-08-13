# NaiveNASlib

[![Build Status](https://travis-ci.com/DrChainsaw/NaiveNASlib.jl.svg?branch=master)](https://travis-ci.com/DrChainsaw/NaiveNASlib.jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl)

NaiveNASlib is a library of functions for mutating computation graphs. It is designed with Neural Architecture Search (NAS) in mind, but can be used for any purpose where doing changes to a model architecture is desired.

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
* Add vertices to the graph
* Remove vertices from the graph
* Add edges to a vertex
* Remove edges to a vertex

For each of the above operations, NaiveNASlib makes the necessary changes to neighboring vertices to ensure that the computation graph is consistent w.r.t dimensions of the activations.

Just to get started, lets create a simple graph for the summation of two numbers:
```julia
in1, in2 = InputVertex.(("in1", "in2"));

computation = CompVertex(+, in1, in2);

graph = CompGraph([in1, in2], computation);

using Test

@test graph(2,3) == 5
```

Now for a more to the point example. The vertex types used above does not contain any information needed to mutate the graph. This might be a sign of OOP damage, but to do so, we need to wrap them in a vertex type which supports mutation.

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

invertex = inputvertex("input", 3)
layer1 = layer(invertex, 4);
layer2 = layer(layer1, 5);

@test [nout(layer1)] == nin(layer2) == [4]

# Lets change the output size of layer1:
Δnout(layer1, -2);

@test [nout(layer1)] == nin(layer2) == [2]
```
As can be seen above, the consequence of changing the output size of `layer1` was that the input size of `layer2` also was changed.

Besides the very simple graph, this mutation was trivial because both layers are of the type `SizeAbsorb`, meaning that a change in number of inputs/outputs does not propagate further in the graph.

Lets do a non-trivial example:
```julia
# When multiplying with a scalar, the output size is the same as the input size.
# This vertex type is said to be size invariant (in lack of better words).
scalarmult(v, f::Integer) = invariantvertex(x -> x .* f, v)

invertex = inputvertex("input", 6);
start = layer(invertex, 6);
split = layer(start, div(nout(invertex) , 3));
joined = conc(scalarmult(split, 2), scalarmult(split,3), scalarmult(split,5), dims=2);
out = start + joined;

@test [nout(invertex)] == nin(start) == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [6]
@test [nout(start), nout(joined)] == nin(out) == [6, 6]

graph = CompGraph(invertex, out)
@test graph((ones(Int, 1,6))) == [78  78  114  114  186  186]

# Ok, lets try to change the size of the vertex "out".
# First we need to realize that we can only change it by integer multiples of 3
# This is because it is connected to "split" through three paths which require nin==nout
# Therefore, any size change to nout of "split" will result in 3 times the change of nin of "out".
# Equivalently, nout of "split" is nin of "out" divided by 3 and nin/nout must be integers.

# We need this information from the layer. Some layers have other requirements
NaiveNASlib.minΔnoutfactor(::SimpleLayer) = 1
NaiveNASlib.minΔninfactor(::SimpleLayer) = 1

@test minΔnoutfactor(out) == minΔninfactor(out) == 3

# Next, we need to define how to mutate our SimpleLayer
NaiveNASlib.mutate_inputs(l::SimpleLayer, newInSize) = l.W = ones(Int, newInSize, size(l.W,2))
NaiveNASlib.mutate_outputs(l::SimpleLayer, newOutSize) = l.W = ones(Int, size(l.W,1), newOutSize)

# In some cases it is useful to hold on to the old graph before mutating
# To do so, we need to define the clone operation for our SimpleLayer
NaiveNASlib.clone(l::SimpleLayer) = SimpleLayer(l.W)
parentgraph = copy(graph)

Δnin(out, 3)

# We didn't touch the input when mutating...
@test [nout(invertex)] == nin(start) == [6]
# Start and joined must have the same size due to elementwise op.
# All three scalarmult vertices are transparent and propagate the size change to split
@test [nout(start)] == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [9]
@test [nout(start), nout(joined)] == nin(out) == [9, 9]

# However, this only updated the mutation metadata, not the actual layer.
# Some reasons for this are shown in the pruning example below
@test graph((ones(Int, 1,6))) == [78  78  114  114  186  186]

# To mutate the graph, we need to apply the mutation:
apply_mutation(graph);

@test graph((ones(Int, 1,6))) == [114  114  114  168  168  168  276  276  276]

# Copy is still intact
@test parentgraph((ones(Int, 1,6))) == [78  78  114  114  186  186]

```

As seen above, things get a little bit out of hand when using:

* Layers which require nin==nout, such as batch normalization and pooling
* Element wise operations
* Concatenation of activations  

The core idea of NaiveNASlib is basically to annotate the type of vertex in the graph so that functions know what is the proper way to deal with the neighboring vertices when mutating a vertex.

This is done through labeling vertices into three major types:
* `SizeAbsorb`: Assumes `nout(v)` and `nin(v)` may change independently. This means that size changes are absorbed by this vertex in the sense they don't propagate further.

* `SizeStack`: Assumes `nout(v) == sum(nin(v))`. This means that size changes propagate forwards (i.e. input -> output and output -> input).

* `SizeInvariant`: Assumes `[nout(v)] == unique(nin(v))`. This means that size changes propagate both forwards and backwards as changing any input size or the output size means all others must change as well.

To use this library to mutate architectures for some neural network library basically means annotating up the above type for each layer type.  

Lets just do a few quick examples of the other use cases.

Prune (and insert) neurons:
```julia
# Some mockup 'batteries' for this example

# First, how to select or add rows or columns to a matrix
# Negative values in selected indicate rows/cols insertion at that index
function select_params(W, selected, dim)
    Wsize = collect(size(W))
    indskeep = repeat(Any[Colon()], 2)
    newmap = repeat(Any[Colon()], 2)

    # The selected indices
    indskeep[dim] = filter(ind -> ind > 0, selected)
    # Where they are 'placed', others will be zero
    newmap[dim] = selected .> 0
    Wsize[dim] = length(newmap[dim])

    newmat = zeros(Int64, Wsize...)
    newmat[newmap...] = W[indskeep...]
    return newmat
end

NaiveNASlib.mutate_inputs(l::SimpleLayer, selected::Vector{<:Integer}) = l.W = select_params(l.W, selected, 1)
NaiveNASlib.mutate_outputs(l::SimpleLayer, selected::Vector{<:Integer}) = l.W = select_params(l.W, selected, 2)

# Return layer just so we can easily look at it
function prunablelayer(in, outsize)
    l = SimpleLayer(reshape(1: nout(in) * outsize, nout(in), :))
    return absorbvertex(l, outsize, in), l
end

# Ok, now lets get down to business!
invertices = inputvertex.(["in1", "in2"], [3,4])
v1, l1 = prunablelayer(invertices[1], 4)
v2, l2 = prunablelayer(invertices[2], 3)
merged = conc(v1, v2, dims=2)
v3, l3 = prunablelayer(merged, 2)
graph = CompGraph(invertices, v3)

@test l1.W ==
[ 1  4  7  10 ;
  2  5  8  11 ;
  3  6  9  12 ]

@test l2.W ==
[ 1  5   9 ;
  2  6  10 ;
  3  7  11 ;
  4  8  12 ]

@test l3.W ==
[ 1   8 ;
  2   9 ;
  3  10 ;
  4  11 ;
  5  12 ;
  6  13 ;
  7  14 ]

# Here is one reason why apply_mutation is needed:
# We want to mutate nin of v3, and we are not sure how this propagates to v1 and v2
# Lets just change the size first and then we see what happens
Δnin(v3, -3)
# Another reason is that it is possible to do several mutations without throwing away
# more information than needed.
# For example, if we had first applied the previous mutation we would have thrown away
# weights for v2 which would then just be replaced by 0s when doing this:
Δnout(v2, 2)

# Lets see that that did...
@test nin(v3) == [6]
@test nout(v1) == 2
@test nout(v2) == 4

# Ok, for v1 we shall remove two output neurons while for v2 we shall add one.
# Changes will propagate to v3 so that the right inputs are chosen
Δnout(v1, [1, 3]) # Keep columns 1 and 3
Δnout(v2, [1,2,3, -1]) # -1 means add a new column

apply_mutation(graph)

@test l1.W ==
[ 1  7 ;
  2  8 ;
  3  9 ]

@test l2.W ==
[ 1  5   9  0 ;
  2  6  10  0 ;
  3  7  11  0 ;
  4  8  12  0 ]

@test l3.W ==
[ 1   8 ;
  3  10 ;
  5  12 ;
  6  13 ;
  7  14 ;
  0   0 ]
```

Add a vertex to a graph:
```julia
invertex = inputvertex("input", 3)
layer1 = layer(invertex, 5)
graph = CompGraph(invertex, layer1)

@test nv(graph) == 2
@test graph(ones(Int, 1, 3)) == [3 3 3 3 3]

# Insert a layer between invertex and layer1
insert!(invertex, vertex -> layer(vertex, nout(vertex)))

@test nv(graph) == 3
@test graph(ones(Int, 1, 3)) == [9 9 9 9 9]
```

Remove a vertex from a graph:
```julia
invertex = inputvertex("input", 3)
layer1 = layer(invertex, 5)
layer2 = layer(layer1, 4)
graph = CompGraph(invertex, layer2)

@test nv(graph) == 3
@test graph(ones(Int, 1, 3)) == [15 15 15 15]

# Remove layer1 and change nin of layer2 from 5 to 3
# Would perhaps have been better to increase nout of invertex, but it is immutable
remove!(layer1)
apply_mutation(graph)

@test nv(graph) == 2
@test graph(ones(Int, 1, 3)) == [3 3 3 3]
```

Add an edge to a vertex:
```julia
invertices = inputvertex.(["input1", "input2"], [3, 2])
layer1 = layer(invertices[1], 4)
layer2 = layer(invertices[2], 4)
add = layer1 + layer2
out = layer(add, 5)
graph = CompGraph(invertices, out)

@test nin(add) == [4, 4]
# Two inputs this time, remember?
@test graph(ones(Int, 1, 3), ones(Int, 1, 2)) == [20 20 20 20 20]

# This graph is not interesting enough for there to be a good showcase for adding a new edge.
# Lets create a new layer which has a different output size just to see how things change
# The only vertex which support more than one input is add
layer3 = layer(invertices[2], 6)
create_edge!(layer3, add)
apply_mutation(graph)

# By default, NaiveNASlib will try to increase the size in case of a mismatch
@test nin(add) == [6, 6, 6]
@test graph(ones(Int, 1, 3), ones(Int, 1, 2)) == [42 42 42 42 42]
```

Remove an edge from a vertex:
```julia
invertex = inputvertex("input", 4)
layer1 = layer(invertex, 3)
layer2 = layer(invertex, 5)
merged = conc(layer1, layer2, layer1, dims=2)
out = layer(merged, 3)
graph = CompGraph(invertex, out)

@test nin(merged) == [3, 5, 3]
@test graph(ones(Int, 1, 4)) == [44 44 44]

remove_edge!(layer1, merged)
apply_mutation(graph)

@test nin(merged) == [5, 6]
@test graph(ones(Int, 1, 4)) == [44 44 44]
```

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
