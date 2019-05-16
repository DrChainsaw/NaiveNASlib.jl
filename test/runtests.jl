using NaiveNASlib
using Test

@testset "NaiveNASlib.jl" begin

@info "Testing computation"

include("vertex.jl")
include("compgraph.jl")

end
