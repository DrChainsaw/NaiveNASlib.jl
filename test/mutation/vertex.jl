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

        @test nout(iv) == 3
        @test nin(iv) == []
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

            @test showstr(show, iv) == "InputVertex(1, outputs=[CompVertex(identity), CompVertex(hcat)])"
            @test showstr(show, cv) == "CompVertex(identity, inputs=[InputVertex(1)], outputs=[CompVertex(hcat)])"
            @test showstr(show, sv) == "CompVertex(hcat, inputs=[InputVertex(1), CompVertex(identity)], outputs=[])"
        end

        @testset "InputSizeVertex" begin
            iv = InputSizeVertex("iv", 3)
            @test showstr(show_less, iv) == "iv"
        end

        @testset "NamedTrait" begin

            v1 = MutationVertex(CompVertex(identity, InputSizeVertex("input1", 3)), NamedTrait(SizeInvariant(), "mv"))

            @test showstr(show_less, v1) == "mv"

            v2 = MutationVertex(CompVertex(+, v1, InputSizeVertex("input2", 3)), NamedTrait(SizeInvariant(), "sv"))

            @test showstr(show, v2) == "MutationVertex(CompVertex(+, inputs=[mv, input2], outputs=[]), NamedTrait(SizeInvariant(), \"sv\"))"
        end

        @testset "Info strings" begin

            v1 = InputSizeVertex("v1", 3)
            v2 = absorbvertex(MatMul(3, 5), v1, traitdecoration=t->NamedTrait(t, "v2"))
            v3 = conc(v1, v2; dims=2, traitdecoration=t->NamedTrait(t, "v3"))
            v4 = invariantvertex(identity, v3; traitdecoration = t-> NamedTrait(t, "v4"))

            @test infostr(NameInfoStr(), v1) == "v1"
            @test infostr(NameInfoStr(), v2) == "v2"
            @test infostr(NameInfoStr(), v3) == "v3"
            @test infostr(NameInfoStr(), v4) == "v4"

            @test infostr(NinInfoStr(), v1) == "N/A"
            @test infostr(NinInfoStr(), v2) == "3"
            @test infostr(NinInfoStr(), v3) == "3, 5"
            @test infostr(NinInfoStr(), v4) == "8"

            @test infostr(SizeInfoStr(), v1) == "nin=[N/A], nout=[3]"
            @test infostr(SizeInfoStr(), v2) == "nin=[3], nout=[5]"
            @test infostr(SizeInfoStr(), v3) == "nin=[3, 5], nout=[8]"
            @test infostr(SizeInfoStr(), v4) == "nin=[8], nout=[8]"

            @test infostr(OutputsInfoStr(), v1) == "[v2, v3]"
            @test infostr(OutputsInfoStr(), v2) == "[v3]"
            @test infostr(OutputsInfoStr(), v3) == "[v4]"
            @test infostr(OutputsInfoStr(), v4) == "[]"

            @test infostr(NameAndIOInfoStr(), v1) == "v1, inputs=[], outputs=[v2, v3]"
            @test infostr(NameAndIOInfoStr(), v2) == "v2, inputs=[v1], outputs=[v3]"
            @test infostr(NameAndIOInfoStr(), v3) == "v3, inputs=[v1, v2], outputs=[v4]"
            @test infostr(NameAndIOInfoStr(), v4) == "v4, inputs=[v3], outputs=[]"

            @test infostr(MutationTraitInfoStr(), v1) == "Immutable()"
            @test infostr(MutationTraitInfoStr(), v2) == "NamedTrait(SizeAbsorb(), v2)"
            @test infostr(MutationTraitInfoStr(), v3) == "NamedTrait(SizeStack(), v3)"
            @test infostr(MutationTraitInfoStr(), v4) == "NamedTrait(SizeInvariant(), v4)"

            @test infostr(MutationSizeTraitInfoStr(), v1) == "Immutable()"
            @test infostr(MutationSizeTraitInfoStr(), v2) == "SizeAbsorb()"
            @test infostr(MutationSizeTraitInfoStr(), v3) == "SizeStack()"
            @test infostr(MutationSizeTraitInfoStr(), v4) == "SizeInvariant()"

            @test infostr(FullInfoStr(), v1) == "v1, inputs=[], outputs=[v2, v3], nin=[N/A], nout=[3], Immutable()"
            @test infostr(FullInfoStr(), v2) == "v2, inputs=[v1], outputs=[v3], nin=[3], nout=[5], SizeAbsorb()"
            @test infostr(FullInfoStr(), v3) == "v3, inputs=[v1, v2], outputs=[v4], nin=[3, 5], nout=[8], SizeStack()"
            @test infostr(FullInfoStr(), v4) == "v4, inputs=[v3], outputs=[], nin=[8], nout=[8], SizeInvariant()"
        end
    end
end
