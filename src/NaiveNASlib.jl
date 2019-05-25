module NaiveNASlib

using LightGraphs
using DataStructures

export
#Interface
AbstractVertex, AbstractMutationVertex, MutationOp, MutationState,

# Vertex
InputVertex, CompVertex, inputs, outputs,

# Computation graph
CompGraph, output!,

# Mutation operations
#Computation
InvSize, IoSize, IoIndices, NoOp, nin, nout, Δnin, Δnout,

# Mutation vertex
base, AbsorbVertex, StackingVertex, InvariantVertex,

# select
select_inputs, select_outputs, select_params

include("vertex.jl")
include("compgraph.jl")

include("mutation/meta.jl")
include("mutation/vertex.jl")
include("mutation/select.jl")

end # module
