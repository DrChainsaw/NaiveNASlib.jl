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
        iv = InputVertex(1)
        cv1 = CompVertex(x -> 3 .* x, AbsorbVertex(iv, InvSize(2)))
        v1 = AbsorbVertex(cv1, IoSize(2, 3))

        @test nin(v1.meta) == [2]
        @test nout(v1.meta) == 3

        @test outputs.(inputs(v1)) == [[v1]]

        Δnin(v1, 2)
        @test nin(v1.meta) == [4]

        Δnout(v1, -2)
        @test nout(v1.meta) == 1

        # Add one vertex and see that change propagates
        cv2 = CompVertex(x -> 3 .+ x, v1)
        v2 = AbsorbVertex(cv2, IoSize(1, 4)) # Note: Size after change above
        @test nout(v1.meta) == nin(v2.meta)[1]
        @test outputs(v1) == [v2]
        @test inputs(v2) == [v1]

        Δnin(v2, 4)
        @test nout(v1.meta) == nin(v2.meta)[1]

        Δnout(v1, -2)
        @test nout(v1.meta) == nin(v2.meta)[1]
    end

end
