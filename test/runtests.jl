using NaiveNASlib
using Test

@testset "NaiveNASlib.jl" begin

@info "Testing computation"

include("vertex.jl")
include("compgraph.jl")

@info "Testing mutation"

include("mutation/meta.jl")
include("mutation/vertex.jl")

end
