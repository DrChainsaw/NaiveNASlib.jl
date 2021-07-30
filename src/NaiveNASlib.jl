module NaiveNASlib

using LightGraphs, MetaGraphs
import Statistics
import Logging
using Logging: LogLevel, @logmsg

import JuMP
using JuMP: @variable, @constraint, @objective, @expression, MOI, MOI.INFEASIBLE, MOI.FEASIBLE_POINT
import Cbc

# Computation graph
export CompGraph, nv, vertices, inputs, outputs, name

# Mutation operations
export nin, nout, Δnin!, Δnout!, Δsize!, relaxed

# Connectivity mutation
export remove!, insert!, create_edge!, remove_edge!

#sugar
export inputvertex, vertex, immutablevertex, absorbvertex, invariantvertex, conc

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

include("api/sugar.jl")
include("api/Advanced.jl")
include("api/Extend.jl")

end # module
