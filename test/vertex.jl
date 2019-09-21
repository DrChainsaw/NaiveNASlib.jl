import NaiveNASlib
import InteractiveUtils:subtypes
using Test

@testset "Basic vertex tests" begin

    @testset "Method contracts $subtype" for subtype in implementations(AbstractVertex)
        @test hasmethod(inputs, (subtype,))
        @test hasmethod(clone, (subtype, Vararg{AbstractVertex}))
    end


    @testset "InputVertex tests" begin
        iv1 = InputVertex(1)
        iv2 = InputVertex("name")

        @test iv1.name == 1
        @test iv2.name == "name"

        @test iv1 == iv1
        @test iv2 == iv2
        @test iv1 != iv2
        @test iv1 == clone(iv1)
        @test iv2 == clone(iv2)
        @test_throws ErrorException clone(iv1, iv1)
    end

    @testset "CompVertex tests" begin

        # Helper function to create vertices with one input
        cv(f, i=InputVertex(1)) = CompVertex(f, i)
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

        @testset "Copy" begin
            iv = InputVertex(1)
            cv_orig = cv(x -> 3x, iv)
            cv_copy = clone(cv_orig, clone(iv, ))

            @test issame(cv_orig, cv_copy)
            @test cv_orig(3) == cv_copy(3)
        end
    end

    @testset "Pretty printing" begin

        @testset "Show CompVertex" begin
            cv = CompVertex(+, InputVertex.(1:2))
            @test showstr(show, cv) == "CompVertex(+), inputs=[InputVertex(1), InputVertex(2)]"
        end

        @testset "Info string CompVertex" begin
            cv = CompVertex(+, InputVertex.(["input1", "input2"]))

            @test infostr(RawInfoStr(), cv) == "CompVertex(+), inputs=[input1, input2]"
            @test infostr(NameInfoStr(), cv) == "CompVertex"
            @test infostr(MutationTraitInfoStr(), cv) == "Immutable()"
            @test infostr(InputsInfoStr(), cv) == "[input1, input2]"
            @test infostr(NameAndInputsInfoStr(), cv) == "CompVertex, inputs=[input1, input2]"
        end

    end
end
