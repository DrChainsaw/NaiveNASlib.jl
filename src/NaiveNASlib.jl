module NaiveNASlib

using LightGraphs
using DataStructures

export
#Interface
AbstractVertex, AbstractMutationVertex, VertexMeta,

# Vertex
InputVertex, CompVertex, inputs, outputs,

# Computation graph
CompGraph, output!,

# Vertex meta
#Computation
InvSize, IoSize, nin, nout, Δnin, Δnout,

# Mutation vertex
base, AbsorbVertex, StackingVertex, InvariantVertex,

# select
select_inputs, select_outputs

include("vertex.jl")
include("compgraph.jl")

include("mutation/meta.jl")
include("mutation/vertex.jl")
include("mutation/select.jl")

end # module
