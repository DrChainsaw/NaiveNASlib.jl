import NaiveNASlib

@testset "Sugar" begin

    @testset "Create input" begin
        @test issame(inputvertex("input", 5), InputSizeVertex("input", 5))
    end

    @testset "Create absorbing" begin
        v = absorbvertex(x -> x * [1 2 3; 4 5 6], 2, inputvertex("in", 3))
        @test nin(v) == [3]
        @test nout(v) == 2
        @test v([1 2]) == [9  12  15]
        v = absorbvertex(identity, 1, inputvertex("in", 1), mutation=IoSize, traitdecoration = t -> NamedTrait(t, "v"))
        @test name(v) == "v"
    end

    @testset "Create stacking" begin
        v = conc(inputvertex.(("in1", "in2"), (1,2))..., dims=1, traitdecoration = t -> NamedTrait(t, "v"))
        @test nin(v) == [1 ,2]
        @test nout(v) == 3
        @test v(1, [2, 3]) == [1,2,3]
        @test name(v) == "v"
    end
end
