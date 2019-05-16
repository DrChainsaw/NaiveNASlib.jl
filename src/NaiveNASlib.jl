module NaiveNASlib

using LightGraphs

export AbstractVertex, InputVertex, CompVertex, CompGraph, inputs, output!

include("vertex.jl")
include("compgraph.jl")

end # module
