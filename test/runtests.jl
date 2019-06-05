using NaiveNASlib
using Test

include("testutil.jl")

@testset "NaiveNASlib.jl" begin

    @info "Testing computation"

    include("vertex.jl")
    include("compgraph.jl")

    @info "Testing mutation"

    include("mutation/op.jl")
    include("mutation/vertex.jl")
    include("mutation/size.jl")
    include("mutation/apply.jl")

end
