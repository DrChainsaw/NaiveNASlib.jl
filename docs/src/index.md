
# Introduction

NaiveNASlib provides a set of easy to use functions to modify the structure of a neural network while making few assumptions on the 
underlying implementation. Apart from the obvious application in neural architecture search, this can also be useful in the context 
of transfer learning and structured pruning (which is a subset of neural architecture search).

Main supported operations:
* Change the input/output size of vertices
* Parameter pruning/insertion (policy excluded)
* Add vertices to the graph
* Remove vertices from the graph
* Add edges to a vertex
* Remove edges to a vertex

For each of the above operations, NaiveNASlib makes the necessary changes to other vertices in the graph to ensure that it 
is consistent w.r.t dimensions of the activations and so it to whatever extent possible represents the same function.

While this is sometimes possible to do manually or through some ad-hoc method, things tend to explode in complexity for 
more complex models. NaiveNASlib comes to the rescue so that you can focus on the actual problem. Any failure to produce a
valid model after mutation warrants an issue!

NaiveNASlib uses [JuMP](https://github.com/jump-dev/JuMP.jl) under the hood to describe not only the size relations, but also the
connections between individual neurons as a Mixed Integer Linear Program (MILP) so that even extremely nested architectures
stay aligned after mutation. 

The price one has to pay is that the model must be explicitly defined as a computation graph in the "language" of this library, 
similar to what some older frameworks using less modern programming languages used to do. In its defense, the main reason anyone
would use this library to begin with is to not have to create computation graphs themselves.
