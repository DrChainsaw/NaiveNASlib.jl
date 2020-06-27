module NaiveNASlib

using LightGraphs, MetaGraphs
using Statistics
using Logging

import JuMP
import JuMP: @variable, @constraint, @objective, @expression, MOI, MOI.INFEASIBLE, MOI.FEASIBLE_POINT
using Cbc

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
export base, InputSizeVertex, OutputsVertex, MutationVertex

# Mutation traits
export trait, MutationTrait, DecoratingTrait, NamedTrait, Immutable, MutationSizeTrait, SizeAbsorb, SizeStack, SizeInvariant, SizeChangeLogger, SizeChangeValidation

# Size util
export minΔnoutfactor, minΔninfactor, minΔnoutfactor_only_for, minΔninfactor_only_for, findterminating, ΔSizeInfo, ΔninSizeInfo, ΔnoutSizeInfo, ΔSizeGraph, ΔninSizeGraph, ΔnoutSizeGraph, Direction, Input, Output, Both, all_in_graph, all_in_Δsize_graph, Δsize, newsizes, neighbours, fullgraph

export AbstractΔSizeStrategy, AbstractJuMPΔSizeStrategy, ΔSizeFailError, ΔSizeFailNoOp, LogΔSizeExec, DefaultJuMPΔSizeStrategy, ΔNout, ΔNoutExact, ΔNoutRelaxed, ΔNin, ΔNinExact, ΔNinRelaxed, AlignNinToNout, Exact, Relaxed

#Selection util
export AbstractSelectionStrategy, LogSelection, LogSelectionFallback, SelectionFail, NoutRevert, SelectDirection, ApplyAfter, AbstractJuMPSelectionStrategy, DefaultJuMPSelectionStrategy, OutSelect, OutSelectExact, OutSelectRelaxed, TruncateInIndsToValid, Δoutputs, solve_outputs_selection

# Connectivity mutation
export remove!, RemoveStrategy, insert!, create_edge!, remove_edge!

# Align size strategies, e.g what to do with sizes of vertices connected to a removed vertex
export AbstractAlignSizeStrategy, IncreaseSmaller, DecreaseBigger, AlignSizeBoth, ChangeNinOfOutputs, AdjustToCurrentSize, FailAlignSizeNoOp, FailAlignSizeError, FailAlignSizeWarn, FailAlignSizeRevert, NoSizeChange, CheckAligned, CheckNoSizeCycle, CheckCreateEdgeNoSizeCycle, PostAlignJuMP, SelectOutputs, ApplyMutation, PostSelectOutputs, PostApplyMutation

# Connect strategies
export AbstractConnectStrategy, ConnectAll, ConnectNone

# apply mutation
export mutate_inputs, mutate_outputs, apply_mutation

#sugar
export inputvertex, vertex, immutablevertex, absorbvertex, invariantvertex, conc, VertexConf, traitconf, mutationconf, outwrapconf

include("vertex.jl")
include("compgraph.jl")
include("prettyprint.jl")

include("mutation/op.jl")
include("mutation/vertex.jl")
include("mutation/graph.jl")
include("mutation/jumpnorm.jl")
include("mutation/size.jl")
include("mutation/apply.jl")
include("mutation/select.jl")

include("mutation/structure.jl")

include("mutation/sugar.jl")


end # module
