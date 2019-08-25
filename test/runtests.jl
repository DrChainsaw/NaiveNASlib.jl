using NaiveNASlib
using Test

include("testutil.jl")

@testset "NaiveNASlib.jl" begin

    @testset "NaiveNASlib.jl legacy" begin

        @info "Testing computation"

        include("vertex.jl")
        include("compgraph.jl")

        @info "Testing mutation"

        include("mutation/op.jl")
        include("mutation/vertex.jl")

        @info "Testing size mutation"

        include("mutation/size.jl")

        @info "Testing index mutation"

        include("mutation/apply.jl")
        include("mutation/select.jl")

        @info "Testing structural mutation"

        include("mutation/structure.jl")

        @info "Testing sugar"

        include("mutation/sugar.jl")
    end
end
