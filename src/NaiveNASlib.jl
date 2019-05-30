module NaiveNASlib

using LightGraphs
using DataStructures

export
#Interface
AbstractVertex, AbstractMutationVertex, MutationOp, MutationState,

# Vertex
InputVertex, CompVertex, inputs, outputs,

# Computation graph
CompGraph, output!,flatten,

# Mutation operations
#State
InvSize, IoSize, IoIndices, NoOp, nin, nout, Δnin, Δnout, clone, op,

# Mutation vertex
base, InputSizeVertex, OutputsVertex, AbsorbVertex, StackingVertex, InvariantVertex,

# select
mutate_inputs, mutate_outputs, apply_mutation

include("vertex.jl")
include("compgraph.jl")

include("mutation/meta.jl")
include("mutation/vertex.jl")
include("mutation/select.jl")


end # module
