module NaiveNASlib

using LightGraphs

export
#Interface
AbstractVertex, AbstractMutationVertex, VertexMeta,

# Vertex
InputVertex, CompVertex, inputs, outputs,

# Computation graph
CompGraph, output!,

# Vertex meta
#Computation
InvSize, IoSize, nin, nout, Δnin, Δnout,

# Mutation vertex
mutate_nin, mutate_nout, base

include("vertex.jl")
include("compgraph.jl")

include("mutation/meta.jl")
include("mutation/vertex.jl")

end # module
