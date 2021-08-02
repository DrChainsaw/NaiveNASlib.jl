module Advanced

# Only exports advanced stuff which one typically can't be arsed to use from the REPl
using Reexport: @reexport
# Traits
@reexport using ..NaiveNASlib: NamedTrait, SizeChangeValidation, SizeChangeLogger, AfterΔSizeTrait

# Size strategies
@reexport using ..NaiveNASlib:  DefaultJuMPΔSizeStrategy, ThrowΔSizeFailError, ΔSizeFailNoOp, LogΔSizeExec, ΔNout, ΔNoutExact, ΔNoutRelaxed, ΔNin, 
                                ΔNinExact, ΔNinRelaxed, AlignNinToNout, TruncateInIndsToValid, WithValueFun, TimeLimitΔSizeStrategy,
                                TimeOutAction, AfterΔSizeCallback, logafterΔsize, validateafterΔsize

# Align size strategies, e.g what to do with sizes of vertices connected to a removed vertex
@reexport using ..NaiveNASlib:  IncreaseSmaller, DecreaseBigger, AlignSizeBoth, ChangeNinOfOutputs, FailAlignSizeNoOp, FailAlignSizeError, 
                                FailAlignSizeWarn, NoSizeChange, CheckAligned, CheckNoSizeCycle, CheckCreateEdgeNoSizeCycle, PostAlign

# Connect strategies
@reexport using ..NaiveNASlib: ConnectAll, ConnectNone

# For elementwise operations
@reexport using ..NaiveNASlib: VertexConf, traitconf, outwrapconf

@reexport using ..NaiveNASlib: RemoveStrategy

@reexport using ..NaiveNASlib: findterminating, all_in_graph, output!, ancestors, descendants, named, logged, validated

end