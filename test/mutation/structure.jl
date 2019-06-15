
@testset "Structure tests" begin

    #Helper functions
    inpt(size, id=1) = InputSizeVertex(id, size)
    av(in, outsize; name="av", comp = identity) = AbsorbVertex(CompVertex(comp, in), IoSize(nout(in), outsize), t -> NamedTrait(t, name))
    sv(in...; name="sv") = StackingVertex(CompVertex(hcat, in...), t -> NamedTrait(t, name))
    iv(in...; name="iv") = InvariantVertex(CompVertex(+, in...), t -> NamedTrait(t, name))

    @testset "Vertex addition"  begin

        @testset "Add to linear graph" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 5, name="v1")
            v2 = av(v1, 4, name="v2")

            @test inputs(v2) != outputs(v1)
            graph = CompGraph(v0, v2)

            @test graph(3) == 3

            insert!(v1, v -> av(v, nout(v), name="vnew1"))

            @test inputs(v2) == outputs(v1)
            vnew1 = inputs(v2)[]
            @test [nout(v1)] == nin(vnew1) == [nout(vnew1)] == nin(v2) == [5]

            @test graph(3) == 3

            @test inputs(vnew1) == [v1]
            @test outputs(vnew1) == [v2]

            # Add two consecutive vertices
            insert!(vnew1, v -> av(av(v, 3, name="vnew2"), nout(v), name="vnew3"))

            @test [inputs(v2)] == outputs.(outputs(vnew1))
            vnew2 = outputs(vnew1)[]
            vnew3 = outputs(vnew2)[]

            @test [nout(vnew1)] == nin(vnew2) == [nout(vnew3)] == nin(v2) == [5]

            @test graph(3) == 3

        end

        @testset "Add to one of many inputs" begin
                v0 = inpt(3, "v0")
                v1 = av(v0, 4, name="v1")
                v2 = av(v0, 5, name="v2")
                v3 = av(v0, 6, name="v3")
                v4 = sv(v1,v2,v3, name="v4")
                v5 = av(v4, 7, name="v5")

                insert!(v2, v -> av(v, nout(v), name="vnew1"))
                vnew1 = outputs(v2)[]

                @test vnew1 != v2
                @test inputs(v4) == [v1, vnew1, v3]

                insert!(v0, v -> av(v, nout(v), name="vnew2"))

                @test length(outputs(v0)) == 1
                vnew2 = outputs(v0)[]

                @test inputs(v1) == inputs(v2) == inputs(v3) == [vnew2]
                @test outputs(vnew2) == [v1, v2, v3]

                insert!(vnew2, v -> av(v, nout(v), name="vnew3"), vouts -> vouts[[1, 3]])

                @test length(outputs(vnew2)) == 2
                @test outputs(vnew2)[1] == v2
                vnew3 = outputs(vnew2)[2]

                @test outputs(vnew3) == [v1, v3]
                @test inputs(v4) == [v1, vnew1, v3]
        end
    end

    @testset "Vertex removal" begin

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

            # Note, input to v1 can not be changed, we must decrease
            # nin of v4 (and v5)
            remove!(v2)
            @test inputs(v4) == [v1, v0, v3]
            @test nin(v5) == [nout(v4)] == [3+4+6]

            #Now lets try without connecting the inputs to v4
            remove!(v1, RemoveStrategy(ConnectNone(), ChangeNinOfOutputs((-nout(v1), missing, missing))))
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

            @test minΔnoutfactor_only_for.(outputs(v2)) == [1]
            @test minΔnoutfactor_only_for.(inputs(v2)) == [3]

            # Impossible to set nout of join to 16 as it is a join of the same vertex 3 times (which is obviously a senseless construct)
            remove!(v2)
            @test nin(v3) == [nout(join)] == [3nout(v1)] == [15]
        end

        @testset "Size constraints" begin

            struct SizeConstraint constraint; end
            NaiveNASlib.minΔnoutfactor(c::SizeConstraint) = c.constraint
            NaiveNASlib.minΔninfactor(c::SizeConstraint) = c.constraint

            @testset "Incompatible size constraints" begin

                v1 = av(inpt(3), 10, name="v1", comp = SizeConstraint(2))
                v2 = av(v1, 5, name = "v2")
                v3 = av(v2, 4, name="v3", comp = SizeConstraint(3))

                @test minΔnoutfactor_only_for.(outputs(v2)) == [3]
                @test minΔnoutfactor_only_for.(inputs(v2)) == [2]

                # Impossible to increase v1 by 5 due to SizeConstraint(3)
                # But also impossible to decrease nin of v3 by 5 due to SizeConstraint(2)
                # However, if we decrease v1 by 2 and increase v3 by 3 we will hit home!
                # Fallback to AlignBoth which does just that
                remove!(v2)
                @test nin(v3) == [nout(v1)] == [8]
            end

            @testset "Incompatible size constraints transparent vertex" begin

                v1 = av(inpt(3), 10, name="v1", comp = SizeConstraint(2))
                v2 = sv(v1, name = "v2")
                v3 = av(v2, 4, name="v3", comp = SizeConstraint(3))

                @test minΔnoutfactor_only_for.(outputs(v2)) == [3]
                @test minΔnoutfactor_only_for.(inputs(v2)) == [2]

                # Size is already aligned due to transparent. Just test that this
                # does not muck things up
                remove!(v2, RemoveStrategy(AlignSizeBoth()))
                @test nin(v3) == [nout(v1)] == [10]
            end
        end

        @testset "Tricky structures" begin

            @testset "Remove residual layers" begin
                v1 = av(inpt(3, "in"), 10, name="v1")
                v2 = av(v1, 3, name="v2")
                v3 = sv(v2, name="v3")
                v4 = av(v3, 10, name="v4")
                v5 = iv(v4, v1, name="v5")
                v6 = av(v5, 4, name="v6")

                remove!(v4)
                @test inputs(v5) == [v3, v1]
                @test nin(v5) == [nout(v3), nout(v1)] == [10, 10]
                @test nin(v6) == [nout(v5)] == [10]

                remove!(v3)
                @test inputs(v5) == [v2, v1]
                @test nin(v5) == [nout(v2), nout(v1)] == [10, 10]
                @test nin(v6) == [nout(v5)] == [10]

                remove!(v2)
                @test inputs(v5) == [v1, v1]
                @test nin(v5) == [nout(v1), nout(v1)] == [10, 10]
                @test nin(v6) == [nout(v5)] == [10]

                v7 = av(v6, 13, name="v7")
                remove!(v6)
                @test nin(v5) == [nout(v1), nout(v1)] == [10, 10]
                @test nin(v7) == [nout(v5)] == [10]

                v8 = av(v7, 3, name="v8")
                remove!(v7)
                @test nin(v5) == [nout(v1), nout(v1)] == [13, 13]
                @test nin(v7) == [nout(v5)] == [13]
            end

            @testset "Remove after transparent fork" begin
                v1 = av(inpt(3, "in"), 5, name="v1")
                p1 = iv(v1, name="p1")
                p2 = iv(v1, name="p2")
                v2 = sv(p1,p2, name="v2")
                v3 = iv(v2, name="v3")
                v4 = av(v3, 12, name="v4")
                v5 = av(v4, 7, name="v5")

                remove!(v4)
                @test inputs(v5) == [v3]
                @test outputs(v3) == [v5]

                @test nin(v5) == [nout(v3)] == [nout(p1) + nout(p2)] == [2 * nout(v1)] == [12]
            end

            @testset "Remove before half transparent resblock" begin
                v1 = av(inpt(2, "in"), 5, name="v1")
                v2 = av(v1, 3, name="v2")
                v3 = iv(v2, name="v3")
                v4 = av(v3, 3, name="v4")
                v5 = iv(v4, name="v5")
                v6 = iv(v3, v4, name="v6")
                v7 = av(v6, 2, name="v7")

                remove!(v5)
                @test inputs(v6) == [v3, v4]
                @test nin(v6) == [nout(v3), nout(v4)] == [3, 3]
            end

            @testset "Remove right before fork" begin
                v1 = av(inpt(3, "in"), 3, name="v1")
                v2 = av(v1, 5, name="v2")
                p1 = iv(v2, name="p1")
                p2₁ = iv(v2, name="p2_1")
                p2₂ = av(p2₁,7, name="p2_2")
                p2₃ = iv(p2₂, name="p2_3")
                v3 = sv(p1, p2₃, name="v3")
                v4 = iv(v3, name="v4")
                v5 = av(v4, 12, name="v5")

                remove!(v2)
                @test inputs(p1) == inputs(p2₁) == [v1]
                @test nin(p1) == nin(p2₁) == [nout(v1)] == [5]
            end
        end
    end
end
