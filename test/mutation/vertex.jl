import NaiveNASlib
import NaiveNASlib:OutputsVertex
import InteractiveUtils:subtypes

@testset "Mutation vertices" begin

    @testset "OutputsVertex" begin
        iv = OutputsVertex(InputVertex(1))

        @test inputs(iv) == []
        @test outputs(iv) == []

        @test issame(iv, clone(iv))

        cv = OutputsVertex(CompVertex(x -> 2x, iv))
        NaiveNASlib.init!(cv, base(cv))

        @test inputs(cv) == [iv]
        @test outputs(iv) == [base(cv)]

        ivc = clone(iv)
        cvc = clone(cv, ivc)
        NaiveNASlib.init!(cvc,base(cvc))

        @test issame(iv, ivc)
    end

    @testset "InputSizeVertex" begin
        iv = InputSizeVertex(InputVertex(1), 3)

        @test nout(iv) == nin(iv) == 3
        @test inputs(iv) == []
        @test outputs(iv) == []

        @test issame(iv, clone(iv))

        cv =  OutputsVertex(CompVertex(x -> 2x, iv))
        NaiveNASlib.init!(cv, base(cv))

        @test inputs(cv) == [iv]
        @test outputs(iv) == [base(cv)]

        ivc = clone(iv)
        cvc = clone(cv, ivc)
        NaiveNASlib.init!(cvc,base(cvc))

        @test issame(iv, ivc)
    end
end
