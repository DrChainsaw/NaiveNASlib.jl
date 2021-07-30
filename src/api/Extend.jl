module Extend

using Reexport: @reexport

# Only exports things one typically dispatches on when extending NaiveNASlib with new functions, e.g. abstract types and traits

# Vertex types
@reexport using ..NaiveNASlib: AbstractVertex, InputVertex, InputSizeVertex, CompVertex, MutationVertex, OutputsVertex

# Mutation traits
@reexport using ..NaiveNASlib: trait, MutationTrait, DecoratingTrait, Immutable, MutationSizeTrait, SizeAbsorb, SizeStack, SizeInvariant

# Size strategies
@reexport using ..NaiveNASlib: AbstractΔSizeStrategy, AbstractJuMPΔSizeStrategy, DecoratingJuMPΔSizeStrategy

@reexport using ..NaiveNASlib: AbstractAlignSizeStrategy
@reexport using ..NaiveNASlib: AbstractConnectStrategy

@reexport using ..NaiveNASlib: base, clone, parselect

end