module NaiveNASlib

import Statistics
import Logging
using Logging: LogLevel, @logmsg

import JuMP
using JuMP: @variable, @constraint, @objective, @expression, MOI, MOI.INFEASIBLE, MOI.FEASIBLE_POINT
import HiGHS
import Functors
using Functors: @functor, functor

import ChainRulesCore
import ChainRulesCore: rrule, rrule_via_ad, RuleConfig, HasReverseMode, NoTangent

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

include("api/vertex.jl")
include("api/size.jl")
include("api/Advanced.jl")
include("api/Extend.jl")

include("chainrules.jl")

end # module
