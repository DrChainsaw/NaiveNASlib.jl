
# Introduction

NaiveNASlib provides a set of easy to use functions to modify the structure of a neural network, or more 
generically, a computation graph. Apart from the obvious application in neural architecture search, this
can also be useful in the context of transfer learning and structured pruning (which is a subset of neural
architecture search).

Main supported operations:
* Change the number of neurons
* Add vertices to the graph
* Remove vertices from the graph
* Add edges to a vertex
* Remove edges to a vertex

For each of the above operations, NaiveNASlib makes the necessary changes to other vertices in the graph to ensure that it 
is consistent w.r.t dimensions of the activations and so it to whatever extent possible represents the same function.

While this is sometimes possible to do manually or through some ad-hoc method, things tend to explode in complexity for 
more complex models. NaiveNASlib comes to the rescue so that you can focus on the actual problem. Any failure to produce a
valid model after mutation warrants an issue!

NaiveNASlib makes quite few assumptions on the underlying implementation which in turn means that it is quite easy to make
use of its capabilities for an existing library. Note that there really isn't anything neural network specific about
NaiveNASlib and it can certainly be used to modify any computation graph. However, most its functionality is dead weight
if there are not at least a handful of operations which require input to have a certain shape along some dimension. 

The price one has to pay is that the model must be explicitly defined as a computation graph in the "language" of this library, 
similar to what some older frameworks using less modern programming languages used to do. In its defense, the main reason anyone
would use this library to begin with is to not have to create computation graphs themselves.

## Reading Guideline

The [Quick Tutorial](@ref) followed by the [Advanced Tutorial](@ref) are written to gradually introduce the ideas
of NaiveNASlib and should serve as a good starting point to tell if this library might be useful to you. 

The [Terminology](@ref) section is meant to clear things up if some recurring word or concept induces uncertainty but 
should be entirely skippable otherwise.

The API reference is split up into the basic API which is the one introduced in the [Quick Tutorial](@ref), the advanced API
which is introduced in [Advanced Tutorial](@ref) and the API for extending NaiveNASlib. Each section is further split up into
categories in an attempt to make it easy to answer "how do I achieve X?"-type questions.

## Under the hood

NaiveNASlib uses [JuMP](https://github.com/jump-dev/JuMP.jl) to describe not only the size relations, but also the
connections between individual neurons as a Mixed Integer Linear Program (MILP). Describing neuron relations with equality 
constraints turned out to give a quite declarative way of formulating the alignment problem and ensures that even extremely 
nested architectures stay aligned after mutation. 

While MILPs are known for being quite difficult it seems like the abundance of equality constraints creates a quite tight
formulation (don't quote me on this though :)) so that even when 10000s of neurons are involved the solution is produced in
sub/few-second time. 
