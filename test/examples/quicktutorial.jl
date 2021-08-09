md"""
# Quick Tutorial

## Construct a very simple graph
Just to get started, lets create a simple graph for the summation of two numbers. 
"""
using NaiveNASlib, Test
md"""
NaiveNASlib uses a special immutable type of vertex to annotate inputs so that one
can be certain that size mutations won't suddenly change the input shape of the model.
"""
@testset "First example" begin #src
in1 = inputvertex("in1", 1)
in2 = inputvertex("in2", 1)

# In this example we could have done without them as we won't have any parameters to 
# change the size of, but lets show things as they are expected to be used.
#
# Create a new vertex which computes the sum of `in1` and `in2`:
add = "add" >> in1 + in2
@test add isa NaiveNASlib.AbstractVertex

# NaiveNASlib lets you do this using [`+`](@ref) which creates a new vertex which sums it inputs. 
# Use `>>` to attach a name to the vertex when using infix operations.
# Naming vertices is completely optional, but it is quite helpful as can be seen below.

# [`CompGraph`](@ref) helps evaluating the whole graph as a function.
graph = CompGraph([in1, in2], add)

# Evaluate the function represented by `graph` by just calling it.
@test graph(2,3) == 5
@test graph(100,200) == 300

# The [`vertices`](@ref) function returns the vertices in topological order.
@test vertices(graph) == [in1, in2, add]
@test name.(vertices(graph)) == ["in1", "in2", "add"]

# [`CompGraph`](@ref)s can be indexed:
@test graph[begin] == graph[1] == in1
@test graph[end] == graph[3] == add
@test graph[begin:end] == vertices(graph)
# This is a bit slow though as it traverses the whole graph each time. It is better to 
# call [`vertices`](@ref) first and then apply the indexing if one needs to do this many times.
end #src

# ## Modify a graph
# Now lets look at how to make use of it to modify the structure of a neural network. Since batteries are excluded, 
# lets first create a tiny neural network library and do the wiring so that NaiveNASlib can work with it.

module TinyNNlib 
    using NaiveNASlib
    ## A simple linear layer
    mutable struct LinearLayer{T}
        W::Matrix{T}
    end
    ## Normally ones uses something like randn here, but this makes output 
    ## in examples easier on the eyes
    LinearLayer(nin, nout) = LinearLayer(ones(Int, nout, nin))
    (l::LinearLayer)(x) = l.W * x

    ## NaiveNASlib needs to know what LinearLayer considers its output and input size
    ## In this case it is the number of rows and columns of the weight matrix
    ## Input size is always a vector since vertices might have multiple inputs
    NaiveNASlib.nin(l::LinearLayer) = [size(l.W, 2)]
    NaiveNASlib.nout(l::LinearLayer) = size(l.W, 1)

    ## We also need to tell NaiveNASlib how to change the size of LinearLayer
    ## The Δsize! function will receive indices to keep from existing weights 
    ## as well as where to insert new indices
    function NaiveNASlib.Δsize!(l::LinearLayer, newins::AbstractVector, newouts::AbstractVector)
        ## newins is a vector of vectors as vertices may have more than one input, 
        ## but LinearLayer has only one
        ## The function NaiveNASlib.parselect can be used to interpret newins and newouts. 
        ## We just need to tell it along which dimensions to apply them.
        l.W = NaiveNASlib.parselect(l.W, 1=>newouts, 2=>newins[])
    end

    ## Helper function which creates a LinearLayer wrapped in an vertex in a computation graph.
    ## This creates a Keras-like API
    linearvertex(in, outsize) = absorbvertex(LinearLayer(nout(in), outsize), in)
    export linearvertex, LinearLayer
end

md"""
There are a handful of other functions one can implement to e.g. provide better defaults and offer other forms of convenience, 
but here we use the bare minimum to get ourselves started. Some examples of this is provided further down.

In practice one might not want to create a whole neural network library from scratch, but rather incorporate NaiveNASlib with an existing library
in a glue package.
It might appear that this will inevitably lead to type-piracy as neither the layer definitions nor the functions (e.g. nout, nin) would 
belong to the glue package. However, one typically wants to wrap the layers in some intermediate type anyways. For instance, 
[NaiveNASflux](https://github.com/DrChainsaw/NaiveNASflux.jl) needs to wrap the layers from Flux in a mutable container as the layers 
themselves are not mutable.

Lets do a super simple example where we make use of the tiny neural network library to create a model and then modify it:
"""

@testset "Second example" begin #src
using .TinyNNlib
invertex = inputvertex("input", 3)
layer1 = linearvertex(invertex, 4)
layer2 = linearvertex(layer1, 5)

# Vertices may be called to execute their computation alone.
# We generally outsource this work to [`CompGraph`](@ref), but now we are trying to illustrate how things work.
batchsize = 2
batch = randn(nout(invertex), batchsize)
y1 = layer1(batch)
@test size(y1) == (nout(layer1), batchsize) == (4, 2)
y2 = layer2(y1)
@test size(y2) == (nout(layer2), batchsize) == (5, 2)

# Lets change the output size of `layer1`. First check the input sizes so we have something to compare to.
@test [nout(layer1)] == nin(layer2) == [4]
@test Δnout!(layer1 => -2) # Returns true if successful
@test [nout(layer1)] == nin(layer2) == [2]

# And now the weight matrices have changed!
#
# The graph is still operational of course but the sizes of the activations have changed.
y1 = layer1(batch)
@test size(y1) == (nout(layer1), batchsize) == (2, 2)
y2 = layer2(y1)
@test size(y2) == (nout(layer2), batchsize) == (5 ,2)
end #src

md"""
## A more elaborate example

As can be seen above, the consequence of changing the output size of `layer1` was that the input size of `layer2` 
also was changed. This is of course required for the computation graph to not throw a dimension mismatch error
when being executed.

Besides the very simple graph, this mutation was trivial because the sizes of the input and output dimensions 
of `LinearLayer`'s parameters can change independently as they are the rows and columns of the weight matrix. 
This is expressed by giving the layers the mutation size trait [`SizeAbsorb`](@ref) in the lingo of NaiveNASlib, 
meaning that a change in number of input/output neurons does not propagate further in the graph.

On to the next example! As hinted before, things can quickly get out of hand when using:

* Layers which require `nin==nout`, e.g. batch normalization and pooling.
* Element wise operations such as activation functions or just element wise arithmetics (e.g `+` used in residual connections).
* Concatenation of activations.  

Lets use a small but non-trivial model including all of the above. We begin by making a helper which creates a vertex which does elementwise scaling of its input:
"""
scalarmult(v, s::Number) = invariantvertex(x -> x .* s, v)
# When multiplying with a scalar, the output size is the same as the input size.
# This vertex type is said to be size invariant (in lack of better words), hence the name [`invariantvertex`](@ref).

@testset "More elaborate example" begin #src
# Ok, lets create the model:
## First a few "normal" layers
invertex = inputvertex("input", 6)
start = linearvertex(invertex, 6)
split = linearvertex(start, nout(invertex) ÷ 3)

## Concatenation means the output size is the sum of the input sizes
joined = conc(scalarmult(split,2), scalarmult(split,3), scalarmult(split,5), dims=1)

## Elementwise addition is of course also size invariant 
out = start + joined

## CompGraph to help us run the whole thing
graph = CompGraph(invertex, out)
@test graph((ones(6)))  == [78, 78, 114, 114, 186, 186]

md"""
Now we have a somewhat complex set of size relations at our hand since the sizes are constrained so that
  1. `start` and `joined` must have the same output size due to element wise addition.
  2. `joined` will always have 3 times the output size of `split` since there are no size absorbing vertices between them.

Modifying this graph manually would of course be manageable (albeit a bit cumbersome) if we created the model
by hand and knew it inside out. When things like the above emerges out of a neural architecture search things 
are less fun though and this is where use of NaiveNASlib will really pay off.

Ok, lets try to increase the size of the vertex `out` by 2. Before we do that, lets have a look at the sizes of the vertices 
in the graph to have something to compare to.
"""
@test [nout(start)] == nin(split) == [3nout(split)] == [sum(nin(joined))] == [nout(out)] == [6]
@test [nout(start), nout(joined)] == nin(out) == [6, 6]

# In many cases it is useful to hold on to the old graph before mutating
parentgraph = deepcopy(graph)

# It is not possible to change the size of `out` by exactly 2 due to `1.` and `2.` above.
# By default, NaiveNASlib warns when this happens and then tries to make the closest possible change.
# If we don't want the warning, we can tell NaiveNASlib to relax and make the closest possible change right away:
@test Δnout!(out => relaxed(2))

@test [nout(start)] == nin(split) == [3nout(split)] == [sum(nin(joined))] == [nout(out)] == [9]
@test [nout(start), nout(joined)] == nin(out) == [9, 9]

# As we can see above, the size change rippled through the graph due to the size relations described above.
# Pretty much every vertex was affected.
#
# Lets evaluate the graph just to verify that we don't get a dimension mismatch error. 
@test graph((ones(6))) == [78, 78, 0, 114, 114, 0, 186, 186, 0]
# `TinyNNlib` uses [`parselect`](@ref) which will insert zeros when size increases by default. 
# This helps the graph maintain the same function after mutation.
# In this case we changed the size of the output layer so we don't have 
# the exact same function though, but hopefully it is clear why e.g. a 
# linear layer after `out` would have made it produce the same output.
#
# Copy is still intact of course.
@test parentgraph((ones(6))) == [78, 78, 114, 114, 186, 186]

md"""
While we still have the complex model in scope, lets show a few more way to change the sizes. See the built in documentation for more information.

It is possible to supply a utility function for telling the utility of each neuron in a vertex. 
NaiveNASlib will prioritize selecting the indices with higher utility.
"""
#
# Prefer high indices:
graphhigh = deepcopy(graph)
@test Δnout!(v -> 1:nout(v), graphhigh[end] => -3)
@test graphhigh((ones(6))) == [42, 0, 60, 0, 96, 0]

# Perfer low indices
graphlow = deepcopy(graph)
@test Δnout!(v -> nout(v):-1:1, graphlow[end] => -3) 
@test graphlow((ones(6))) == [78, 78, 114, 114, 186, 186]


# A common approach when doing structured pruning is to prefer neurons with high magnitude.
# Here is how to set that as the default for LinearLayer.
# This is something one should probably implement in `TinyNNlib` instead.
using Statistics: mean
NaiveNASlib.defaultutility(l::LinearLayer) = mean(abs, l.W, dims=2)

graphhighmag = deepcopy(graph)
@test Δnout!(graphhighmag[end] => -3) 
@test graphhighmag((ones(6))) == [78, 78, 114, 114, 186, 186]

# In many NAS applications one wants to apply random mutations to the graph.
# When doing so, one might end up in situations like this:
badgraphdecinc = deepcopy(graph)
v1, v2 = badgraphdecinc[[3, end]] # Imagine selecting these at random
@test Δnout!(v1 => relaxed(-2))
@test Δnout!(v2 => 6)
## Now we first deleted a bunch of weights, then we added new :(
@test badgraphdecinc((ones(6))) ==  [42, 0, 0, 60, 0, 0, 96, 0, 0]

# In such cases, it might be better to supply all wanted changes in one go and let 
# NaiveNASlib try to come up with a decent compromise.
goodgraphdecinc = deepcopy(graph)
v1, v2 = goodgraphdecinc[[3, end]]
@test Δnout!(v1 => relaxed(-2), v2 => 3) # Mix relaxed and exact size changes freely
@test goodgraphdecinc((ones(6))) == [78.0, 78.0, 6.0, 0.0, 108.0, 114.0, 6.0, 6.0, 180.0, 180.0, 0.0, 0.0]

# It is also possible to change the input direction, but it requires specifying a size change for each input and is generally not recommended due to this.
graphΔnin = deepcopy(graph)
v1, v2 = graphΔnin[end-1:end]
## Use missing to signal "don't care"
@test Δnin!(v1 => (3, relaxed(2), missing), v2 => relaxed((1,2)))
@test nin(v1) == [6, 6, 6] # Sizes are tied to nout of split so they all have to be equal
@test nin(v2) == [18, 18] # Sizes are tied due to elementwise addition

# A popular pruning strategy is to just remove the x% of params with lowest utility.
# This can be done by just not putting any size requirements and assign negative utility.
graphprune40 = deepcopy(graph)
Δsize!(graphprune40) do v
    utility = NaiveNASlib.defaultutility(v)
    ## We make some strong assumptions on weight distribution here for breviety :)
    return utility .- 0.4mean(utility)
end
@test nout.(vertices(graphprune40)) == [6, 6, 2, 2, 2, 2, 6, 6]
## Compare to original:
@test nout.(vertices(graph))        == [6, 9, 3, 3, 3, 3, 9, 9]

end #src

@testset "Weight modification example" begin #src

# ## Closer look at how weights are modified
# Here we take a closer look at how the weight matrices are changed.

function vertexandlayer(in, outsize)
    nparam = nout(in) * outsize
    l = LinearLayer(collect(reshape(1:nparam, :, nout(in))))
    ## Return vertex and wrapped layer just so we can easily look at it
    return absorbvertex(l, in), l
end

# Make a simple model:
invertices = inputvertex.(["in1", "in2"], [3,4])
v1, l1 = vertexandlayer(invertices[1], 4)
v2, l2 = vertexandlayer(invertices[2], 3)
merged = conc(v1, v2, dims=1)
v3, l3 = vertexandlayer(merged, 2)
graph = CompGraph(invertices, v3)

# These weights might look a bit odd, but they are of course intialized to 
# make it easier to spot what has changed after size change below.
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

# Now, lets decrease `v1` by 1 and force `merged` to retain its size
# which in turn forces `v2` to grow by 1.
# Assign utility 10 to neurons 1 and 3 of `v2` and 1 for all other neurons in the model. 
@test Δnout!(v2 => -1, merged => 0) do v
    v == v2 ? [10, 1, 10] : 1
end

# `v1` got a new row of parameters at the end:
@test l1.W ==
[ 1  5   9 ;
  2  6  10 ;
  3  7  11 ;
  4  8  12 ;
  0  0   0 ]

# `v2` chose to drop its middle row as it was the output neuron with lowest utility:
@test l2.W ==
[ 1 4 7 10 ;
  3 6 9 12 ]

# `v3` dropped the second to last column (which is aligned to the middle row of `v2`).
# and got new parameters in column 5 (which is aligned to the last row of `v1`):
@test l3.W ==
[  1  3  5  7  0   9  13 ;
   2  4  6  8  0  10  14 ]
end #src


# ## Other modifications
# Lets just do a few quick examples of the other types of modifications.

# ### Add a vertex

# Using [`insert!`](@ref):

@testset "Add vertex example" begin #src
invertex = inputvertex("input", 3)
layer1 = linearvertex(invertex, 5)
graph = CompGraph(invertex, layer1)

## nvertices(g) is shortcut for length(vertices(g))
@test nvertices(graph) == 2
@test graph(ones(3)) == [3,3,3,3,3]

## Insert a layer between invertex and layer1
@test insert!(invertex, vertex -> linearvertex(vertex, nout(vertex))) # True if success

@test nvertices(graph) == 3
@test graph(ones(3)) == [9, 9, 9, 9, 9]
end #src

# ### Remove a vertex

# Using [`remove!`](@ref):
@testset "Remove vertex example" begin #src
invertex = inputvertex("input", 3)
layer1 = linearvertex(invertex, 5)
layer2 = linearvertex(layer1, 4)
graph = CompGraph(invertex, layer2)

@test nvertices(graph) == 3
@test graph(ones(3)) == [15, 15, 15, 15]

## Remove layer1 and change nin of layer2 from 5 to 3
## Would perhaps have been better to increase nout of invertex, but it is immutable
@test remove!(layer1) # True if success

@test nvertices(graph) == 2
@test graph(ones(3)) == [3, 3, 3, 3]
end #src

# ### Add an edge

# Using [`create_edge!`](@ref):
@testset "Add edge example" begin
invertices = inputvertex.(["input1", "input2"], [3, 2])
layer1 = linearvertex(invertices[1], 4)
layer2 = linearvertex(invertices[2], 4)
add = layer1 + layer2
out = linearvertex(add, 5)
graph = CompGraph(invertices, out)

@test nin(add) == [4, 4]
## Two inputs to this graph!
@test graph(ones(3), ones(2)) == [20, 20, 20, 20, 20]

## This graph is not interesting enough for there to be a good showcase for adding a new edge.
## Lets create a new layer which has a different output size just to see how things change
## The only vertex which support more than one input is add
layer3 = linearvertex(invertices[2], 6)
@test create_edge!(layer3, add) # True if success

## NaiveNASlib will try to increase the size in case of a mismatch by default
@test nin(add) == [6, 6, 6]
@test graph(ones(3), ones(2)) == [28, 28, 28, 28, 28] 
end #src

# ### Remove an edge

# Using [`remove_edge!`](@ref):
@testset "Remove edge example" begin #src
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
end #src
#

