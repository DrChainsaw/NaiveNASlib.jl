import NaiveNASlib: AbstractMutationVertex, Δnin, Δnout, base, AbsorbVertex
import InteractiveUtils:subtypes

using Test

@testset "Mutation vertices" begin

    @testset "Method contracts" begin
        for subtype in subtypes(AbstractMutationVertex)
            @info "test method contracts for $subtype"
            @test hasmethod(Δnin,  (subtype, Vararg{Integer}))
            @test hasmethod(Δnout, (subtype, Integer))
            @test hasmethod(base, (subtype,))
        end
    end

    @testset "AbsorbVertex" begin
        v = AbsorbVertex(InputVertex(1), IoSize(2, 3))

        @test nin(v.meta) == [2]
        @test nout(v.meta) == 3
        
        Δnin(v, 2)
        @test nin(v.meta) == [4]

        Δnout(v, -2)
        @test nout(v.meta) == 1
    end

end
