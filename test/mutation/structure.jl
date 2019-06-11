
@testset "Structure tests" begin
    @testset "Vertex removal" begin

        #Helper functions
        inpt(size, id=1) = InputSizeVertex(id, size)
        av(in, outsize; name="av", comp = identity) = AbsorbVertex(CompVertex(comp, in), IoSize(nout(in), outsize), t -> NamedTrait(t, name))
        sv(in...; name="sv") = StackingVertex(CompVertex(hcat, in...), t -> NamedTrait(t, name))
        iv(in...; name="iv") = InvariantVertex(CompVertex(+, in...), t -> NamedTrait(t, name))

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

            # Test that it is possible to remove vertex without any outputs
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

        @testset "Incompatible size factor" begin
            v1 = av(inpt(3), 5, name="v1")
            p1 = iv(v1, name="p1")
            p2 = iv(v1, name="p2")
            p3 = iv(v1, name="p3")
            join = sv(p1,p2,p3, name="join")
            v2 = av(join, 16, name = "v2") # 16 is not divisible by 3!
            v3 = av(v2, 4, name="v3")

            @test minΔnoutfactor.(outputs(v2)) == [1]
            @test minΔnoutfactor.(inputs(v2)) == [3]

            # Impossible to set nout of join to 16 as it is a join of the same vertex 3 times (which is obviously a senseless construct)
            remove!(v2)
            @test nin(v3) == [nout(join)] == [3nout(v1)] == [15]
        end

        @testset "Incompatible size constraints" begin

            struct SizeConstraint constraint; end
            NaiveNASlib.minΔnoutfactor(c::SizeConstraint) = c.constraint
            NaiveNASlib.minΔninfactor(c::SizeConstraint) = c.constraint

            v1 = av(inpt(3), 10, name="v1", comp = SizeConstraint(2))
            v2 = av(v1, 5, name = "v2")
            v3 = av(v2, 4, name="v3", comp = SizeConstraint(3))

            @test minΔnoutfactor.(outputs(v2)) == [3]
            @test minΔnoutfactor.(inputs(v2)) == [2]

            # Impossible to increase v1 by 5 due to SizeConstraint(3)
            # But also impossible to decrease nin of v3 by 5 due to SizeConstraint(2)
            # However, if we decrease v1 by 2 and increase v3 by 3 we will hit home!
            # Fallback to AlignBoth which does just that
            remove!(v2)
            @test nin(v3) == [nout(v1)] == [8]
        end
    end
end
