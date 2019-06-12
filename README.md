# NaiveNASlib

[![Build Status](https://travis-ci.com/DrChainsaw/NaiveNASlib.jl.svg?branch=master)](https://travis-ci.com/DrChainsaw/NaiveNASlib.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/NaiveNASlib.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/NaiveNASlib-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl)

Work in progress! Limited functionality (and bugs, but there will always be bugs).

NaiveNASlib is a library of functions for mutating computation graphs in order to support Neural Architecture Search (NAS).

It is "batteries excluded" in the sense that it is independent of both neural network implementation and search policy implementation.

Its only contribution to this world is some help with the sometimes annoyingly complex procedure of changing an existing neural network architecture into a new, similar yet different, neural network architecture.

## Basic usage

```
Pkg.add("https://github.com/DrChainsaw/NaiveNASlib.jl")
```

The price one has to pay is that the computation graph must be explicitly defined in the "language" of this library, similar to what some older frameworks using less modern programming languages used to do (in its defense, the sole reason anyone would use this library to begin with is to not have to create computation graphs themselves).

Disclaimer: Syntax is a bit clumsy. It will hopefully change into something more terse before wip status is removed.

Just to get started, lets create a simple graph for the summation of two numbers:
```
in1, in2 = InputVertex.(("in1", "in2"));

computation = CompVertex(+, in1, in2);

graph = CompGraph([in1, in2], [computation]);

graph(2,3)
5
```

Now for a more to the point example. The vertex types used above does not contain any information needed to mutate the graph. This might be a sign of OOP damage, but in order to do so, we need to wrap them in a vertex type which supports mutation.

```
# First we need something to mutate. Batteries excluded, remember?
mutable struct SimpleLayer
    W
end
SimpleLayer(nin, nout) = SimpleLayer(ones(nin,nout))
(l::SimpleLayer)(x) = x * l.W

# Helper function which creates a mutable layer
layer(in, outsize) = MutationVertex(CompVertex(SimpleLayer(nout(in), outsize), in), IoSize(nout(in), outsize), SizeAbsorb())

input = InputSizeVertex("input", 3)
layer1 = layer(input, 4);
layer2 = layer(layer1, 5);

nout(layer1)
4

nin(layer2)
1-element Array{Integer,1}:
 4

#Lets change the output size of layer1:

Δnout(layer1, -2);

nout(layer1)
2

nin(layer2)
1-element Array{Integer,1}:
 2
```
As can be seen above, the consequence of changing the output size of ```layer1``` was that the input size of ```layer2``` also was changed.

This mutation was trivial because both layers are of the type ```SizeAbsorb```, meaning that a change in number of inputs/outputs does not propagate further in the graph. Things do however go out of hand when using:

* Layers which require nin==nout such as batch normalization
* Element wise operations
* Concatenation of activations  

The core idea of NaiveNASlib is basically to annotate the type of vertex in the graph so that functions know what is the proper way to deal with the neighbouring vertices when mutating a vertex.

Main supported use cases:
* Change the input/output size of vertex
* Parameter pruning (policy excluded)
* Remove a vertex from the graph

To be added:
* Add a vertex to the graph
* Add/remove edges to/from vertex

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
