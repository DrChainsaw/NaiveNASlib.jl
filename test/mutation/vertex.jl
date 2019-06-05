import NaiveNASlib
import NaiveNASlib:OutputsVertex
import InteractiveUtils:subtypes

@testset "Mutation vertices" begin

    @testset "Method contracts" begin
        for subtype in subtypes(AbstractMutationVertex)
            @info "\ttest method contracts for $subtype"
            @test hasmethod(nin,  (subtype,))
            @test hasmethod(nout, (subtype,))
            @test hasmethod(Δnin,  (subtype, Integer))
            @test hasmethod(Δnout, (subtype, Integer))
            @test hasmethod(base, (subtype,))
            @test hasmethod(outputs, (subtype,))
        end
    end

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

    @testset "Vertex removal" begin

        #Helper functions
        inpt(size, id=1) = InputSizeVertex(InputVertex(1), size)
        av(in, outsize) = AbsorbVertex(CompVertex(identity, in), IoSize(nout(in), outsize))
        sv(in...) = StackingVertex(CompVertex(hcat, in...))
        rb(in...) = InvariantVertex(CompVertex(+, in...))

        @testset "Remove from linear graph" begin
            v0 = inpt(3)
            v1 = av(v0, 5)
            v2 = av(v1, 4)
            v3 = av(v2,6)

            remove!(v2)
            @test inputs(v3) == [v1]
            @test outputs(v1) == [v3]
            @test nin(v3) == [nout(v1)] == [5]

            # Note, input to v1 can not be changed, we must decrease
            # nin of v3
            remove!(v1)
            @test inputs(v3) == [v0]
            @test outputs(v0) == [v3]
            @test nin(v3) == [nout(v0)] == [3]
        end

        @testset "Remove one of many inputs" begin
            v0 = inpt(3)
            v1 = av(v0, 4)
            v2 = av(v0, 5)
            v3 = av(v0, 6)
            v4 = sv(v1,v2,v3)
            v5 = av(v4, 7)

            remove!(v2)
            @test inputs(v4) == [v1, v0, v3]
            @test nin(v5) == [nout(v4)] == [3+4+6]

            #Now lets try without connecting the inputs to v4
            remove!(v1, RemoveStrategy(ConnectNone(), ChangeNinOfOutputs(-nout(v1))))
            @test inputs(v4) == [v0, v3]
            @test nin(v5) == [nout(v4)] == [3+6]
        end

        @testset "Remove one of many outputs" begin
            v0 = inpt(3)
            v1 = av(v0, 4)
            v2 = av(v1, 5)
            v3 = av(v1, 6)
            v4 = av(v1, 7)
            v5 = av(v2, 8)

            remove!(v2)
            @test outputs(v1) == [v5, v3, v4]
            @test nin(v5) == nin(v3) == nin(v4) == [nout(v1)] == [5]

            # Test that it is possible to remove vertex without and outputs
            remove!(v3)
            @test outputs(v1) == [v5, v4]
            @test nin(v5) == nin(v4) == [nout(v1)] == [6]
        end

        @testset "Hidden immutable" begin
            v0 = inpt(3)
            v1 = sv(v0)
            v2 = av(v1, 4)
            v3 = av(v2, 5)

            #Danger! Must realize that size of v1 can not be changed!
            remove!(v2)
            @test outputs(v1) == [v3]
            @test inputs(v3) == [v1]
            @test nin(v3) == [nout(v1)] == [3]
        end
    end
end
