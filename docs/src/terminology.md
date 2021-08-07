# Terminology

NaiveNASlib tries to use standard graph and neural network terminology to whatever extent possible. Here is a short summary of the 
important concepts and what they mean to NaiveNASlib. This section is best read after the [Quick Tutorial](@ref) to make 
it somewhat more concrete.

## Vertex

Vertices are the fundamental unit which NaiveNASlib works with when changing the structure of a graph. A vertex
can be queried for both input and output vertices as well as its current input and output size (see [Neuron](@ref) below). 

Most vertices in the graph wrap a function which from the point of view of NaiveNASlib are the primitives of the computation 
graph. NaiveNASlib does not have a firm opinion on what functions are considered primitive though. One vertex in the graph can
wrap a single layer (or something even simpler) while another can wrap a whole computation graph.

As seen in the [Quick Tutorial](@ref), how to adjust the input/output sizes of the vertices in the graph depends on the 
computation.

NaiveNASlib uses traits to classify three basic types of vertices so that it does not have to implement rules for every
possible primitive. The core idea of NaiveNASlib is basically to annotate the type of vertex in the graph so that it knows 
what is the proper way to deal with the neighboring vertices when mutating a vertex.

This is done through labeling vertices into three major types:
* `SizeAbsorb`: Assumes `nout(v)` and `nin(v)` may change independently. This means that size changes are absorbed by this
    vertex in the sense they don't propagate further. Most typical neural network layers with parameter arrays fall into this
    category.

* `SizeStack`: Assumes `nout(v) == sum(nin(v))`. This means that size changes propagate forwards (i.e. input -> output and
    output -> input). The main operation in this category is concatenation of activations. 

* `SizeInvariant`: Assumes `[nout(v)] == unique(nin(v))`. This means that size changes propagate both forwards and backwards
    as changing any input size or the output size means all others must change as well. In this category we find typically find
  element wise operations, but also normalization and pooling operations tend to fall into this category. 

NaiveNASlib also uses the term `SizeTransparent` to denote the latter two (i.e any vertex which is not `SizeAbsorb`).
To use this library to mutate architectures for some neural network library basically means annotating up the above type for 
each layer type and connect parameter dimensions to input and output sizes.

Note one typically does not need to interact with the traits when just using NaiveNASlib and they are not exported by default.
The functions for [Vertex Creation](@ref) attaches the proper trait to the vertex when creating it.

While the above covers a substantial set of operations, it is possible to implement special rules for individual computations
as well.

## Edge

Contrary to more general graph frameworks, edges in NaiveNASlib are implicit in the sense that each vertex stores its input 
and output vertices. Edges are primarily used when evaluating the graph as a function as well as when formulating the 
constraints for keeping the graph size aligned.

## Neuron

Neuron is the name NaiveNASlib uses for the indices of the relevant dimension of the arrays passed between vertices. For example,
if one vertex `v` takes a vector of size `N` as input and returns a vector of size `M` it has `N` input neurons
and `M` output neurons. This is synonymous with saying that `v` has input size `N` and output size `M` and `nin(v) == N` and
`nout(v) == M`.

When changing input or output size `v` will be given indices of the neurons to keep as well as
where to insert new neurons. NaiveNASlib has then made sure that other vertices have gotten indices so that all remaining 
neurons stay connected to the same neurons they were connected to previously. See [Closer look at how weights are modified](@ref)
for a concrete example.

NaiveNASlib does not have the ability to figure out what input and output sizes a function wrapped in a vertex have by itself,
so this must be provided by the implementation. See [`nout`](@ref) and [`nin`](@ref).

Examples of how to determine the input and output size for common layer types:

* Fully connected layers: The size of the non-batch dimension, typically the rows/columns of the weight matrix.
* Recurrent layers: The size of the dimension which is neither batch, nor time. 
* Convolutional layers: The number of input/output channels. 