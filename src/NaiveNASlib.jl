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

# Connectivity mutation
remove!, RemoveStrategy,

# Align size strategies, e.g what to do with sizes of vertices connected to a removed vertex
AbstractAlignSizeStrategy, IncreaseSmaller, DecreaseBigger, ChangeNinOfOutputs, Fail,

# Connect strategies
AbstractConnectStrategy, ConnectAll, ConnectNone,

# apply mutation
mutate_inputs, mutate_outputs, apply_mutation

include("vertex.jl")
include("compgraph.jl")

include("mutation/op.jl")
include("mutation/vertex.jl")
include("mutation/size.jl")
include("mutation/apply.jl")

include("mutation/structure.jl")



end # module
