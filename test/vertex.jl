@testset "Basic vertex tests" begin

    @testset "Method contracts $subtype" for subtype in implementations(AbstractVertex)
        @test hasmethod(inputs, (subtype,))
        @test hasmethod(Functors.functor, (subtype,))
        @test hasmethod(nin, (subtype,))
        @test hasmethod(nout, (subtype,))
        @test hasmethod(name, (subtype,))
        @test hasmethod(NaiveNASlib.nameorrepr, (subtype,))
    end

    @testset "InputVertex tests" begin
        iv1 = InputVertex(1)
        iv2 = InputVertex("name")

        @test iv1.name == 1
        @test iv2.name == "name"

        @test iv1 == iv1
        @test iv2 == iv2
        @test iv1 != iv2
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

        @testset "Copy CompVertex with $label" for (label, cfun) in (
            (deepcopy, deepcopy),
            ("fmap", g -> Functors.fmap(identity, g))
        )
            iv = InputVertex(1)
            cv_orig = cv(x -> 3x, iv)
            cv_copy = cfun(cv_orig)

            @test issame(cv_orig, cv_copy)
            @test cv_orig(3) == cv_copy(3)
        end
    end

    @testset "Pretty printing" begin

        @testset "Show CompVertex" begin
            cv = CompVertex(+, InputVertex.(1:2))
            @test showstr(show, cv) == "CompVertex(+, inputs=[InputVertex(1), InputVertex(2)])"
        end
    end
end
