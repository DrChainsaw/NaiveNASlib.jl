import NaiveNASlib
import InteractiveUtils:subtypes
using Test


@testset "Computation vertex" begin

    @testset "Method contracts" begin
        for subtype in subtypes(AbstractVertex)
            @info "test method contracts for $subtype"
            @test hasmethod(inputs, (subtype,))
        end
    end

    # Helper function to create vertices with one input
    cv(f) = CompVertex(f, InputVertex(1))
    @testset "Scalar operations" begin
        @test cv(x -> 5)(2) == 5
        @test cv(x -> 5x)(2) == 10
        @test cv(sum)([2,3]) == 5
        @test cv(+)(3,4) == 7
        @test cv(vcat)(5,6) == [5,6]
    end

    @testset "Array operations" begin
        @test cv(x -> x .* 5)(ones(2,3)) == [5 5 5; 5 5 5]
        @test cv(x -> [3 4] * x)(ones(2,3)) == [7.0 7.0 7.0]
    end

    # @testset "FixComp operations" begin
    #     fc(f) = cv(FixComp(f, 1, 1))
    #     @test typeof(fc(identity)) == CompVertex{FixComp}
    #     @test fc(x -> 5x)(2) == 10
    #     @test fc(+)(2,3) == 5
    #     @test fc(sum)([2,3]) == 5
    #     @test fc(vcat)("test", "passed") == ["test", "passed"]
    #     @test fc(*)("test", " ", "passed") == "test passed"
    # end

end
