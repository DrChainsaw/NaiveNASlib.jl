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
InvSize, IoSize, InvIndices, IoIndices, NoOp, nin, nout, Δnin, Δnout, clone, op,

# Mutation vertex
base, InputSizeVertex, OutputsVertex, AbsorbVertex, StackingVertex, InvariantVertex,

# apply mutation
mutate_inputs, mutate_outputs, apply_mutation

include("vertex.jl")
include("compgraph.jl")

include("mutation/op.jl")
include("mutation/vertex.jl")
include("mutation/apply.jl")


end # module
