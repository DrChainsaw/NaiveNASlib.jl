module NaiveNASlib

using PrecompileTools

import Statistics
import Logging
using Logging: LogLevel, @logmsg

import JuMP
using JuMP: @variable, @constraint, @objective, @expression, MOI, MOI.INFEASIBLE, MOI.FEASIBLE_POINT
import HiGHS
import Functors
using Functors: @functor, functor

# Computation graph
export CompGraph, nvertices, vertices, findvertices, inputs, outputs, name

# Vertex size operations
export nin, nout, Δnin!, Δnout!, Δsize!, relaxed

# Connectivity mutation
export remove!, insert!, create_edge!, remove_edge!

# Create vertices
export inputvertex, immutablevertex, absorbvertex, invariantvertex, conc

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


@recompile_invalidations begin
    # Haven't checked whether this does invalidate, but it does syntax punning (which I do regret a bit today)
    # on some Base methods, e.g. +
    include("api/vertex.jl")
end
include("api/size.jl")
include("api/Advanced.jl")
include("api/Extend.jl")

end # module
