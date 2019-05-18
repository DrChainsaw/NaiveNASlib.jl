import NaiveNASlib: VertexMeta, InvSize, IoSize, nin, nout, Δnout, Δnin
import InteractiveUtils:subtypes

using Test

@testset "Vertex mutation metadata" begin

    @testset "Method contracts" begin
        for subtype in subtypes(VertexMeta)
            @info "test method contracts for $subtype"
            @test hasmethod(nin, (subtype,))
            @test hasmethod(nout, (subtype,))
            @test hasmethod(Δnin, (subtype,Vararg{Integer}))
            @test hasmethod(Δnout, (subtype, Integer))
        end
    end

    @testset "InvSize" begin
        size = InvSize(3)
        @test nin(size) == 3
        @test nout(size) == 3

        Δnin(size, 2)
        @test nin(size) == 5
        @test nout(size) == 5

        Δnout(size, -3)
        @test nin(size) == 2
        @test nout(size) == 2

        @test_throws AssertionError Δnin(size, 1,2)
    end

    @testset "IoSize" begin
        size = IoSize(3, 4)
        @test nin(size) == [3]
        @test nout(size) == 4

        Δnin(size, -1)
        @test nin(size) == [2]
        @test nout(size) == 4

        Δnout(size, 2)
        @test nin(size) == [2]
        @test nout(size) == 6

        size = IoSize([2, 3, 4], 5)
        @test nin(size) == [2, 3, 4]

        Δnin(size, 1,-2, 3)
        @test nin(size) == [3, 1 ,7]
    end

end
