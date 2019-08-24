module NaiveNASlib

using LightGraphs, MetaGraphs
using DataStructures
using Statistics
# For solving diophantine equations which pop up here and there when mutating size under constraints
using AbstractAlgebra
using LinearAlgebra
using Logging

# For solving pesky entangled neuron select problems. To be moved to NaiveNASlib if things work out
import JuMP
import JuMP: @variable, @constraint, @objective, @expression, MOI, MOI.INFEASIBLE, MOI.FEASIBLE_POINT
using Cbc
using Juniper
using Ipopt

#Interface
export AbstractVertex, AbstractMutationVertex, MutationOp, MutationState

# Vertex
export InputVertex, CompVertex, inputs, outputs

# Information strings
export infostr, name, RawInfoStr, NameInfoStr, InputsInfoStr, OutputsInfoStr, SizeInfoStr, MutationTraitInfoStr, ComposedInfoStr, NameAndInputsInfoStr, NinInfoStr, NoutInfoStr, NameAndIOInfoStr, FullInfoStr,MutationSizeTraitInfoStr

# Computation graph
export CompGraph, SizeDiGraph, output!,flatten, nv, vertices

# Mutation operations
#State
export InvSize, IoSize, InvIndices, IoIndices, NoOp, IoChange, nin, nout, Δnin, Δnout, clone, op, in_inds, out_inds, nin_org, nout_org

# Mutation vertex
export base, InputSizeVertex, OutputsVertex, AbsorbVertex, StackingVertex, InvariantVertex, MutationVertex

# Mutation traits
export trait, MutationTrait, DecoratingTrait, NamedTrait, Immutable, MutationSizeTrait, SizeAbsorb, SizeStack, SizeInvariant, SizeChangeLogger, SizeChangeValidation

# Size util
export minΔnoutfactor, minΔninfactor, minΔnoutfactor_only_for, minΔninfactor_only_for, findterminating, ΔSizeInfo, ΔninSizeInfo, ΔnoutSizeInfo, ΔSizeGraph, ΔninSizeGraph, ΔnoutSizeGraph, Direction, Input, Output

export ΔNoutLegacy, ΔNinLegacy, AbstractJuMPSizeStrategy, ΔSizeFail, DefaultJuMPΔSizeStrategy, ΔNoutExact

#Selection util
export AbstractSelectionStrategy, LogSelection, SelectionFail, NoutRevert, AbstractJuMPSelectionStrategy, NoutExact, NoutRelaxSize, NoutMainVar, validouts, select_outputs

# Connectivity mutation
export remove!, RemoveStrategy, insert!, create_edge!, remove_edge!

# Align size strategies, e.g what to do with sizes of vertices connected to a removed vertex
export AbstractAlignSizeStrategy, IncreaseSmaller, DecreaseBigger, AlignSizeBoth, ChangeNinOfOutputs, AdjustToCurrentSize, FailAlignSizeError, FailAlignSizeWarn, FailAlignSizeRevert, NoSizeChange, CheckAligned, CheckNoSizeCycle

# Connect strategies
export AbstractConnectStrategy, ConnectAll, ConnectNone

# apply mutation
export mutate_inputs, mutate_outputs, apply_mutation

#sugar
export inputvertex, vertex, immutablevertex, absorbvertex, invariantvertex, conc, VertexConf, traitconf, mutationconf, outwrapconf

include("vertex.jl")
include("compgraph.jl")

include("mutation/op.jl")
include("mutation/vertex.jl")
include("mutation/size.jl")
include("mutation/apply.jl")
include("mutation/select.jl")

include("mutation/structure.jl")

include("mutation/sugar.jl")


end # module
