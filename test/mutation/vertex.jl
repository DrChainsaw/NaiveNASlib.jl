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

    @testset "Pretty printing" begin
        import NaiveNASlib: show_less

        @testset "OutputsVertex" begin
            iv = OutputsVertex(InputVertex(1))
            cv = OutputsVertex(CompVertex(identity, iv))
            NaiveNASlib.init!(cv, base(cv))
            sv = OutputsVertex(CompVertex(hcat, iv, cv))
            NaiveNASlib.init!(sv, base(sv))

            @test showstr(show, iv) == "InputVertex(1), outputs=[CompVertex(identity), CompVertex(hcat)]"
            @test showstr(show, cv) == "CompVertex(identity), inputs=[InputVertex(1)], outputs=[CompVertex(hcat)]"
            @test showstr(show, sv) == "CompVertex(hcat), inputs=[InputVertex(1), CompVertex(identity)], outputs=[]"
        end

        @testset "InputSizeVertex" begin
            iv = InputSizeVertex("iv", 3)
            @test showstr(show_less, iv) == "iv"
        end

        @testset "NamedTrait" begin

            v1 = InvariantVertex(CompVertex(identity, InputSizeVertex("input1", 3)),  t -> NamedTrait(t, "mv"))

            @test showstr(show_less, v1) == "mv"

            v2 = InvariantVertex(CompVertex(+, v1, InputSizeVertex("input2", 3)), t -> NamedTrait(t, "sv"))

            @test showstr(show, v2) == "MutationVertex(CompVertex(+), inputs=[mv, input2], outputs=[], NoOp(), NamedTrait(SizeInvariant(), \"sv\"))"
        end

    end
end
