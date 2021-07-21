module NaiveNASlib

using LightGraphs, MetaGraphs
import Statistics
import Logging
import Logging: LogLevel, @logmsg

import JuMP
import JuMP: @variable, @constraint, @objective, @expression, MOI, MOI.INFEASIBLE, MOI.FEASIBLE_POINT
import Cbc

#Interface
export AbstractVertex

# Vertex
export InputVertex, CompVertex, inputs, outputs

# Information strings
export infostr, name, RawInfoStr, NameInfoStr, InputsInfoStr, OutputsInfoStr, SizeInfoStr, MutationTraitInfoStr, ComposedInfoStr, NameAndInputsInfoStr, NinInfoStr, NoutInfoStr, NameAndIOInfoStr, FullInfoStr,MutationSizeTraitInfoStr

# Computation graph
export CompGraph, SizeDiGraph, output!, flatten, nv, vertices

#Size query and mutation
export nin, nout, Δnin!, Δnout!, clone, relaxed

# Mutation vertex
export base, InputSizeVertex, OutputsVertex, MutationVertex

# Mutation traits
export trait, MutationTrait, DecoratingTrait, NamedTrait, Immutable, MutationSizeTrait, SizeAbsorb, SizeStack, SizeInvariant, SizeChangeLogger, SizeChangeValidation

# Size util
export minΔnoutfactor, minΔninfactor, minΔnoutfactor_only_for, minΔninfactor_only_for, findterminating, ΔSizeInfo, ΔSizeGraph, ΔninSizeGraph, ΔnoutSizeGraph, Direction, Input, Output, Both, all_in_graph, all_in_Δsize_graph, Δsize!, newsizes, neighbours, fullgraph

export AbstractΔSizeStrategy, AbstractJuMPΔSizeStrategy, ThrowΔSizeFailError, ΔSizeFailNoOp, LogΔSizeExec, DefaultJuMPΔSizeStrategy, ΔNout, ΔNoutExact, ΔNoutRelaxed, ΔNin, ΔNinExact, ΔNinRelaxed, AlignNinToNout

#Selection util
export SelectDirection, TruncateInIndsToValid, WithValueFun

# Connectivity mutation
export remove!, RemoveStrategy, insert!, create_edge!, remove_edge!

# Align size strategies, e.g what to do with sizes of vertices connected to a removed vertex
export AbstractAlignSizeStrategy, IncreaseSmaller, DecreaseBigger, AlignSizeBoth, ChangeNinOfOutputs, AdjustToCurrentSize, FailAlignSizeNoOp, FailAlignSizeError, FailAlignSizeWarn, FailAlignSizeRevert, NoSizeChange, CheckAligned, CheckNoSizeCycle, CheckCreateEdgeNoSizeCycle, PostAlign

# Connect strategies
export AbstractConnectStrategy, ConnectAll, ConnectNone

#sugar
export inputvertex, vertex, immutablevertex, absorbvertex, invariantvertex, conc, VertexConf, traitconf, outwrapconf

include("vertex.jl")
include("compgraph.jl")
include("prettyprint.jl")

include("mutation/vertex.jl")
include("mutation/graph.jl")

include("mutation/jumpnorm.jl")
include("mutation/sizestrategies.jl")

include("mutation/size.jl")
include("mutation/select.jl")
include("mutation/structure.jl")

include("mutation/sugar.jl")


end # module
