module NaiveNASlib

using LightGraphs
using DataStructures
using Statistics

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
base, InputSizeVertex, OutputsVertex, AbsorbVertex, StackingVertex, InvariantVertex, MutationVertex,

# Mutation traits
trait, MutationTrait, NamedTrait, Immutable, MutationSizeTrait, SizeAbsorb, SizeStack, SizeInvariant,

# Size util
minΔnoutfactor, minΔninfactor, findterminating,

# Connectivity mutation
remove!, RemoveStrategy,

# Align size strategies, e.g what to do with sizes of vertices connected to a removed vertex
AbstractAlignSizeStrategy, IncreaseSmaller, DecreaseBigger, AlignLcm, ChangeNinOfOutputs, FailAlignSize,

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
