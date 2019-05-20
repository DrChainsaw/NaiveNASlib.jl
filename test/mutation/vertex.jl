import NaiveNASlib
import NaiveNASlib:OutputsVertex
import InteractiveUtils:subtypes

using Test

@testset "Mutation vertices" begin

    @testset "Method contracts" begin
        for subtype in subtypes(AbstractMutationVertex)
            @info "test method contracts for $subtype"
            @test hasmethod(nin,  (subtype,))
            @test hasmethod(nout, (subtype,))
            @test hasmethod(Δnin,  (subtype, Vararg{Integer}))
            @test hasmethod(Δnout, (subtype, Integer))
            @test hasmethod(base, (subtype,))
            @test hasmethod(outputs, (subtype,))
        end
    end

    @testset "AbsorbVertex" begin
        iv = AbsorbVertex(InputVertex(1), InvSize(2))
        v1 = AbsorbVertex(CompVertex(x -> 3 .* x, iv), IoSize(2, 3))

        @test nin(v1) == [2]
        @test nout(v1) == 3

        @test outputs.(inputs(v1)) == [[v1]]

        Δnin(v1, 2)
        @test nin(v1) == [4]

        Δnin(v1, -3)
        @test nin(v1) == [1]

        Δnout(v1, -2)
        @test nout(v1) == 1

        # Add one vertex and see that change propagates
        v2 = AbsorbVertex(CompVertex(x -> 3 .+ x, v1), IoSize(1, 4)) # Note: Size after change above
        @test nout(v1) == nin(v2)[1]
        @test outputs(v1) == [v2]
        @test inputs(v2) == [v1]

        Δnin(v2, 4)
        @test nout(v1) == nin(v2)[1] == 5

        Δnout(v1, -2)
        @test nout(v1) == nin(v2)[1] == 3

        # Fork of v1 into a new vertex
        v3 = AbsorbVertex(CompVertex(identity, v1), IoSize(3, 2))
        @test outputs(v1) == [v2, v3]
        @test inputs(v3) == inputs(v2) == [v1]

        Δnout(v1, -2)
        @test nout(v1) == nin(v2)[1] == nin(v3)[1] == 1

        Δnin(v3, 3)
        @test nout(v1) == nin(v2)[1] == nin(v3)[1] == 4

        Δnin(v2, -2)
        @test nout(v1) == nin(v2)[1] == nin(v3)[1] == 2
    end

    @testset "TransparentVertex" begin

        @testset "TransparentVertex 1 to 1" begin
            iv = AbsorbVertex(InputVertex(1), InvSize(2))
            tv = TransparentVertex(CompVertex(identity, iv))
            io = AbsorbVertex(CompVertex(identity, tv), InvSize(2))

            @test outputs.(inputs(io)) == [[io]]
            @test outputs.(inputs(tv)) == [[tv]]
            @test outputs(iv) == [tv]

            Δnout(iv, 3)
            @test nout(iv) == nin(tv)[1] == nout(tv) == nin(io) == 5

            Δnin(io, -2)
            @test nout(iv) == nin(tv)[1] == nout(tv) == nin(io) == 3
        end

        @testset "TransparentVertex 2 inputs" begin
            # Try with two inputs to TransparentVertex
            iv1 = AbsorbVertex(InputVertex(1), InvSize(2))
            iv2 = AbsorbVertex(InputVertex(2), InvSize(3))
            tv = TransparentVertex(CompVertex(hcat, iv1, iv2))
            io1 = AbsorbVertex(CompVertex(identity, tv), InvSize(5))

            @test inputs(tv) == [iv1, iv2]
            @test outputs.(inputs(tv)) == [[tv], [tv]]
            @test outputs(iv1) == [tv]
            @test outputs(iv2) == [tv]

            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == 5

            Δnout(iv2, -2)
            @test nin(io1) == nout(tv) == sum(nin(tv)) == nout(iv1) + nout(iv2)  ==  3

            Δnin(io1, 3)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == 6

            Δnout(iv1, 4)
            Δnin(io1, -8)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == 2
            @test nout(iv1) == nout(iv2) == 1

            #Add another output
            io2 = AbsorbVertex(CompVertex(identity, tv), InvSize(2))

            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 2

            Δnout(iv1, 3)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 5

            Δnin(io2, -2)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 3

            Δnin(io1, 3)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 6
        end
    end
    @testset "InvariantVertex" begin

        @testset "InvariantVertex 1 to 1" begin
            #Behaviour is identical to TransparentVertex in this case
            iv = AbsorbVertex(InputVertex(1), InvSize(2))
            inv = InvariantVertex(CompVertex(identity, iv))
            io = AbsorbVertex(CompVertex(identity, inv), InvSize(2))

            @test outputs.(inputs(io)) == [[io]]
            @test outputs.(inputs(inv)) == [[inv]]
            @test outputs(iv) == [inv]

            Δnout(iv, 3)
            @test nout(iv) == nin(inv)[1] == nout(inv) == nin(io) == 5

            Δnin(io, -2)
            @test nout(iv) == nin(inv)[1] == nout(inv) == nin(io) == 3

        end

        @testset "InvariantVertex 2 inputs" begin
            # Try with two inputs a InvariantVertex
            iv1 = AbsorbVertex(InputVertex(1), InvSize(3))
            iv2 = AbsorbVertex(InputVertex(2), InvSize(3))
            inv = InvariantVertex(CompVertex(hcat, iv1, iv2))
            io1 = AbsorbVertex(CompVertex(identity, inv), InvSize(3))

            @test inputs(inv) == [iv1, iv2]
            @test outputs.(inputs(inv)) == [[inv], [inv]]
            @test outputs(iv1) == [inv]
            @test outputs(iv2) == [inv]

            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == 3

            Δnout(iv2, -2)
            @test nin(io1) == nout(inv) == nin(inv)[1] == nin(inv)[2]  == nout(iv1) == nout(iv2)  ==  1

            Δnin(io1, 3)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == 4

            #Add another output
            io2 = AbsorbVertex(CompVertex(identity, inv), InvSize(4))

            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == nin(io2) == 4

            Δnout(iv1, -3)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == nin(io2) == 1

            Δnin(io2, 2)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == nin(io2) == 3

            Δnout(iv2, 3)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == nin(io2) == 6
        end

    end

end
