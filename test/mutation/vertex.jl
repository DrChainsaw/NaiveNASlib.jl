@testset "Mutation vertices" begin
    using Functors: fmap

    @testset "OutputsVertex with $label" for (label, cfun) in (
        (deepcopy, deepcopy),
        ("fmap", v -> (vc = fmap(deepcopy, v); NaiveNASlib.init!(vc, base(vc)); return vc)),
    )
        iv = OutputsVertex(InputVertex(1))

        @test inputs(iv) == []
        @test outputs(iv) == []

        @test issame(iv, cfun(iv))

        cv = OutputsVertex(CompVertex(x -> 2x, iv))
        NaiveNASlib.init!(cv, base(cv))

        @test inputs(cv) == [iv]
        @test outputs(iv) == [base(cv)]

        cvc = cfun(cv)

        ivc = inputs(cvc)[]
        @test issame(iv, ivc)
    end

    @testset "InputSizeVertex with $label" for (label, cfun) in (
        (deepcopy, deepcopy),
        ("fmap", v -> (vc = fmap(deepcopy, v); if vc isa OutputsVertex NaiveNASlib.init!(vc, base(vc)) end; return vc)),
    )
        iv = InputSizeVertex(InputVertex(1), 3)

        @test nout(iv) == 3
        @test nin(iv) == []
        @test inputs(iv) == []
        @test outputs(iv) == []

        @test issame(iv, cfun(iv))

        cv =  OutputsVertex(CompVertex(x -> 2x, iv))
        NaiveNASlib.init!(cv, base(cv))

        @test inputs(cv) == [iv]
        @test outputs(iv) == [base(cv)]

        cvc = cfun(cv)
 
        ivc = inputs(cvc)[]
        @test issame(iv, ivc)
    end

    @testset "Fixed size trait" begin
        using NaiveNASlib: vertex
        struct FixedSizeDummy end
        NaiveNASlib.nout(::FixedSizeDummy) = 5
        NaiveNASlib.nin(f::FixedSizeDummy) = [nout(f)]

        @testset "Fixed Size $stype" for stype in (SizeStack, SizeInvariant)
            iv = inputvertex("in", nout(FixedSizeDummy()) - 1)
            v1 = vertex(FixedSizeTrait(stype()), FixedSizeDummy(), iv)

            @test nout(v1) == nout(FixedSizeDummy())
            @test nin(v1) == nin(FixedSizeDummy())
        end
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
            using NaiveNASlib: MutationVertex
            v1 = MutationVertex(CompVertex(identity, InputSizeVertex("input1", 3)), NamedTrait("mv", SizeInvariant()))

            @test showstr(show_less, v1) == "mv"

            v2 = MutationVertex(CompVertex(+, v1, InputSizeVertex("input2", 3)), NamedTrait("sv",SizeInvariant()))

            @test showstr(show, v2) == "MutationVertex(CompVertex(+, inputs=[mv, input2], outputs=[]), NamedTrait(\"sv\", SizeInvariant()))"
        end
    end
end
