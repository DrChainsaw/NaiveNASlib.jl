import NaiveNASlib
import NaiveNASlib:OutputsVertex
import InteractiveUtils:subtypes

using Test

@testset "Mutation vertices" begin

    @testset "Method contracts" begin
        for subtype in subtypes(AbstractMutationVertex)
            @info "test method contracts for $subtype"
            @test hasmethod(Δnin,  (subtype, Vararg{Integer}))
            @test hasmethod(Δnout, (subtype, Integer))
            @test hasmethod(base, (subtype,))
            @test hasmethod(outputs, (subtype,))
        end
    end

    @testset "AbsorbVertex" begin
        iv = AbsorbVertex(InputVertex(1), InvSize(2))
        v1 = AbsorbVertex(CompVertex(x -> 3 .* x, iv), IoSize(2, 3))

        @test nin(v1.meta) == [2]
        @test nout(v1.meta) == 3

        @test outputs.(inputs(v1)) == [[v1]]

        Δnin(v1, 2)
        @test nin(v1.meta) == [4]

        Δnin(v1, -3)
        @test nin(v1.meta) == [1]

        Δnout(v1, -2)
        @test nout(v1.meta) == 1

        # Add one vertex and see that change propagates
        v2 = AbsorbVertex(CompVertex(x -> 3 .+ x, v1), IoSize(1, 4)) # Note: Size after change above
        @test nout(v1.meta) == nin(v2.meta)[1]
        @test outputs(v1) == [v2]
        @test inputs(v2) == [v1]

        Δnin(v2, 4)
        @test nout(v1.meta) == nin(v2.meta)[1] == 5

        Δnout(v1, -2)
        @test nout(v1.meta) == nin(v2.meta)[1] == 3

        # Fork of v1 into a new vertex
        v3 = AbsorbVertex(CompVertex(identity, v1), IoSize(3, 2))
        @test outputs(v1) == [v2, v3]
        @test inputs(v3) == inputs(v2) == [v1]

        Δnout(v1, -2)
        @test nout(v1.meta) == nin(v2.meta)[1] == nin(v3.meta)[1] == 1

        Δnin(v3, 3)
        @test nout(v1.meta) == nin(v2.meta)[1] == nin(v3.meta)[1] == 4

        Δnin(v2, -2)
        @test nout(v1.meta) == nin(v2.meta)[1] == nin(v3.meta)[1] == 2
    end

end
