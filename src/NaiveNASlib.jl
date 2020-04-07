module NaiveNASlib

using LightGraphs, MetaGraphs
using Statistics
using Logging

import JuMP
import JuMP: @variable, @constraint, @objective, @expression, MOI, MOI.INFEASIBLE, MOI.FEASIBLE_POINT
import Cbc

#Interface
export AbstractVertex, AbstractMutationVertex

# Vertex
export InputVertex, CompVertex, inputs, outputs

# Information strings
export infostr, name, RawInfoStr, NameInfoStr, InputsInfoStr, OutputsInfoStr, SizeInfoStr, MutationTraitInfoStr, ComposedInfoStr, NameAndInputsInfoStr, NinInfoStr, NoutInfoStr, NameAndIOInfoStr, FullInfoStr, MutationSizeTraitInfoStr

# Computation graph
export CompGraph, SizeDiGraph, output!, ancestors, nv, vertices

# Mutation operations
#State
export nin, nout, Δnin, Δnout, clone, in_inds, out_inds, nin_org, nout_org

# Mutation vertex
export InputSizeVertex, MutationVertex

# Mutation traits
export trait, MutationTrait, DecoratingTrait, NamedTrait, Immutable, MutationSizeTrait, SizeAbsorb, SizeStack, SizeInvariant, SizeChangeLogger, SizeChangeValidation

# Size util
export minΔnoutfactor, minΔninfactor, minΔnoutfactor_only_for, minΔninfactor_only_for, findterminating, ΔSizeInfo, ΔninSizeInfo, ΔnoutSizeInfo, ΔSizeGraph, ΔninSizeGraph, ΔnoutSizeGraph, Direction, Input, Output, Both, all_in_graph, all_in_Δsize_graph, Δsize, newsizes, neighbours, fullgraph

export AbstractΔSizeStrategy, AbstractJuMPΔSizeStrategy, ΔSizeFailError, ΔSizeFailNoOp, LogΔSizeExec, DefaultJuMPΔSizeStrategy, ΔNout, ΔNoutExact, ΔNoutRelaxed, ΔNin, ΔNinExact, ΔNinRelaxed, AlignNinToNout, Exact, Relaxed

#Selection util
export AbstractSelectionStrategy, LogSelection, LogSelectionFallback, SelectionFail, NoutRevert, SelectDirection, ApplyAfter, AbstractJuMPSelectionStrategy, DefaultJuMPSelectionStrategy, OutSelect, OutSelectExact, OutSelectRelaxed, Δoutputs, solve_outputs_selection

# Connectivity mutation
export remove!, RemoveStrategy, insert!, create_edge!, remove_edge!

# Align size strategies, e.g what to do with sizes of vertices connected to a removed vertex
export AbstractAlignSizeStrategy, IncreaseSmaller, DecreaseBigger, AlignSizeBoth, ChangeNinOfOutputs, AdjustToCurrentSize, FailAlignSizeError, FailAlignSizeWarn, FailAlignSizeRevert, NoSizeChange, CheckAligned, CheckNoSizeCycle, CheckCreateEdgeNoSizeCycle, PostAlignJuMP, SelectOutputs, ApplyMutation, PostSelectOutputs, PostApplyMutation

# Connect strategies
export AbstractConnectStrategy, ConnectAll, ConnectNone

# apply mutation
export apply_mutation

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
