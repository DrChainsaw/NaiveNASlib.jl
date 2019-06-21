
@testset "Structure tests" begin

    @testset "Solve systems of linear diophantine equations" begin
        import NaiveNASlib: solve_lin_dio_eq

        V, T, dof = NaiveNASlib.solve_lin_dio_eq(hcat(4), [8])
        @test dof == 0
        @test V == hcat(1) # its a 1x1 matrix
        @test T == [2]
        @test 4 * V * T == [8]

        V, T, dof = NaiveNASlib.solve_lin_dio_eq(hcat([2, 5]), [4, 10])
        @test dof == 0
        @test V == hcat(1)
        @test T == [2]

        A = [2 3 5]
        V, T, dof = NaiveNASlib.solve_lin_dio_eq(A, [7])
        @test dof == 2
        @test A * V * vcat(T, zeros(Int, dof)) == [7]

        A = [1 2 3; 3 2 1; 2 4 6]; # Note: A[1,:] == 2*A[3,:]
        @test ismissing(NaiveNASlib.solve_lin_dio_eq(A, [1,2,3])) # B[1] != B[3]
        @test ismissing(NaiveNASlib.solve_lin_dio_eq(A, [1,2,2])) # Fails D[n] == 0 ∀ n > k
        V, T, dof = NaiveNASlib.solve_lin_dio_eq(A, [1,3,2])
        @test dof == 1
        @test A * V * vcat(T, 666ones(Int, dof)) == [1,3,2]

        # Underdetermined
        A = [1 2 3 4; 5 6 7 8]
        V,T,dof = NaiveNASlib.solve_lin_dio_eq(A , [2,6])

        @test dof == 2
        @test A * V * vcat(T, -37ones(Int, dof)) == [2, 6]

        # Overdetermined
        A = [1 2; 3 4; 5 6]
        V,T,dof = NaiveNASlib.solve_lin_dio_eq([1 2; 3 4; 5 6], [3,3,3])
        @test dof == 0
        @test A * V * T == [3,3,3]
    end

    @testset "Align sizes" begin
        import NaiveNASlib: alignfor

        @testset "Simple 0 to 1 cases" begin
            @test alignfor(1, missing, [1], [1]) == [0, 0]
            @test alignfor(1, missing, [2], [1]) == [0, -1]
            @test ismissing(alignfor(2, missing, [1], [2]))
            @test alignfor(5+2*7, missing, [5], [7]) == [0, 14]
        end

        @testset "Simple 0 to 2 cases" begin
            @test alignfor(1, missing, [1, 2], [1, 1]) == [0, 0, -1]
            @test alignfor(1, missing, [2, 3], [1, 2]) == [0, -1, -2]
            @test ismissing(alignfor(2, missing, [2, 3], [2, 2]))
            @test alignfor(5+2*7, missing, [5, 7], [7, 6]) == [0, 14, 12]
        end

        @testset "Simple 1 to 1 cases" begin
            @test alignfor(1, 1, [1], [1]) == [0, 0]
            @test alignfor(1, 1, [2], [1]) == [1, 0]
            @test alignfor(1, 2, [2], [1]) == [2, 1]
            @test alignfor(2, 1, [1], [2]) == [1, 2]
            @test alignfor(1, 3, [5], [7]) == [18, 14]
            @test alignfor(5, 7, [19], [missing]) == [14, 0]
        end

        @testset "Simple 1 to 2 cases" begin
            @test alignfor(1, 1, [1, 1], [1, 1]) == [0,0,0]
            @test alignfor(1, 1, [2, 2], [1, 1]) == [1,0,0]
            @test alignfor(1, 2, [2, 3], [1, 1]) == [2,1,0]
            @test alignfor(2, 3, [5, 7], [11, 13]) == [135, 132, 130]
            @test alignfor(3, 3, [9, 18], [9, missing]) == [15, 9, 0]
            @test alignfor(3, 2, [7, 7], [missing, missing]) == [4, 0, 0]
        end

        @testset "Simple 1 to 3 cases" begin
            @test alignfor(1, 1, [1, 1, 1], [1, 1, 1]) == [0,0,0,0]
            @test alignfor(1, 1, [2, 2, 2], [1, 1, 1]) == [1,0,0,0]
            @test alignfor(1, 2, [2, 3, 4], [1, 1, 1]) == [4,3,2,1]
            @test alignfor(2, 3, [5, 7, 11], [13, 17, 19]) == [3201, 3198, 3196, 3192]
            @test alignfor(2, 2, [4, 8, 16], [3, 4, missing]) == [14, 12, 8, 0]
            @test alignfor(2, 3, [4, 8, 8], [2, missing, missing]) == [6, 4, 0, 0]
        end

        @testset "Simple 2 to 1 cases" begin
            @test alignfor([1, 0], [1, 1], [1], [1]) == [0, 0, 0]
            @test alignfor([1, 1], [1, 1], [1], [1]) == [0, -1, 0]
            @test alignfor([2, 3], [5, 7], [11], [13]) == [55, -49, 0]
            @test alignfor([2, 3], [5, 7], [11], [missing]) == [-15, 21, 0]
            @test alignfor([2, 5], [7, missing], [8], [1]) == [0, 0, -1]
            @test alignfor([3, 2], [2, 3], [13], [3]) == [2, 3, -3]
        end

        @testset "Simple 2 to 3 cases" begin
            @test alignfor([1, 0], [1, 1], [1, 1, 1], [1, 1, 1]) == [0,0,0,0,0]
            @test alignfor([1, 0], [1, 1], [2, 2, 2], [1, 1, 1]) == [0,1,0,0,0]
            @test alignfor([1, 2], [1, 1], [2, 2, 2], [1, 1, 1]) == [0,-1,0,0,0]
            @test alignfor([1, 3], [2, 5], [2, 3, 4], [3, 5, 7]) == [-206, 255, 51, 50, 49]
            @test alignfor([2, 3], [3, 5], [11, 8, 8], [3, missing, missing]) == [3, 0, -3, 0, 0]
        end

        @testset "Edge cases" begin

            @test ismissing(alignfor(1, 2, [0,0], [0, 0]))

            # Size differences causes negative Δs, but heuristic finds positive solution
            @test alignfor(50, 2, [3,5], [3, 5]) == [10, 57, 55]

            # Large size differences causes negative Δs, large absolute values stops search for positive solutions
            @test alignfor(2, 3, [3, 12345], [5, 7]) == [1521, 1520, -10822]
            @test alignfor(54321, 2, [3, 5], [7, 11]) == [-5542, 48776, 48774]
        end
    end

    #Helper functions
    inpt(size, id=1) = InputSizeVertex(id, size)
    av(in, outsize; name="av", comp = identity) = AbsorbVertex(CompVertex(comp, in), IoSize(nout(in), outsize), t -> NamedTrait(t, name))
    sv(in...; name="sv") = StackingVertex(CompVertex(hcat, in...), t -> NamedTrait(t, name))
    iv(in...; name="iv") = InvariantVertex(CompVertex(+, in...), t -> NamedTrait(t, name))

    @testset "Edge addition" begin

        @testset "Add to single output stacking" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 5, name="v1")
            v2 = av(v0, 4, name="v2")
            v3 = sv(v1, name = "v3")
            v4 = av(v3, 3, name="v4")
            v5 = av(v2, 2, name="v5")

            @test inputs(v3) == [v1]
            create_edge!(v2, v3)

            @test inputs(v3) == [v1, v2]
            @test nin(v4) == [nout(v3)] == [nout(v1) + nout(v2)] == [9]

            @test outputs(v2) == [v5, v3]
            @test inputs(v5) == [v2]
            @test nin(v5) == [nout(v2)] == [4]
        end

        @testset "Add duplicate to single output stacking" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 5, name="v1")
            v2 = av(v0, 4, name="v2")
            v3 = sv(v1, v1, v2, name = "v3")
            v4 = av(v3, 3, name="v4")
            v5 = av(v2, 2, name="v5")

            @test inputs(v3) == [v1, v1, v2]
            create_edge!(v1, v3)

            @test inputs(v3) == [v1, v1, v2, v1]
            @test nin(v4) == [nout(v3)] == [3nout(v1) + nout(v2)] == [19]

            @test outputs(v2) == [v3, v5]
            @test inputs(v5) == [v2]
            @test nin(v5) == [nout(v2)] == [4]
        end

        @testset "Add immutable to single output stacking" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 5, name="v1")
            v2 = sv(v1, name = "v2")
            v3 = av(v2, 3, name="v3")

            @test inputs(v2) == [v1]
            create_edge!(v0, v2)

            @test inputs(v2) == [v1, v0]
            @test nin(v3) == [nout(v2)] == [nout(v1) + nout(v0)] == [8]

            @test outputs(v0) == [v1, v2]
        end

        @testset "Add to single output invariant" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 5, name="v1")
            v2 = av(v0, 4, name="v2")
            v3 = iv(v1, name = "v3")
            v4 = av(v3, 3, name="v4")
            v5 = av(v2, 2, name="v5")

            @test inputs(v3) == [v1]
            create_edge!(v2, v3)

            @test inputs(v3) == [v1, v2]
            @test nin(v4) == [nout(v3)] == [nout(v1)] == [nout(v2)] == [5]

            @test outputs(v2) == [v5, v3]
            @test inputs(v5) == [v2]
            @test nin(v5) == [nout(v2)] == [5]
        end

        @testset "Add duplicate to single output invariant" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 4, name="v1")
            v2 = av(v0, 4, name="v2")
            v3 = iv(v1, v1, v2, name = "v3")
            v4 = av(v3, 3, name="v4")
            v5 = av(v2, 2, name="v5")

            @test inputs(v3) == [v1, v1, v2]
            create_edge!(v1, v3)

            @test inputs(v3) == [v1, v1, v2, v1]
            @test nin(v4) == [nout(v3)] == [nout(v1)] == [nout(v2)] == [4]

            @test outputs(v2) == [v3, v5]
            @test inputs(v5) == [v2]
            @test nin(v5) == [nout(v2)] == [4]
        end

        @testset "Add immutable to single output invariant" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 5, name="v1")
            v2 = iv(v1, name = "v2")
            v3 = av(v2, 3, name="v3")

            @test inputs(v2) == [v1]
            create_edge!(v0, v2)

            @test inputs(v2) == [v1, v0]
            @test nin(v3) == [nout(v2)] == [nout(v1)] == [nout(v0)] == [3]

            @test outputs(v0) == [v1, v2]
        end

        @testset "Size constraints" begin

            struct SizeConstraint constraint; end
            NaiveNASlib.minΔnoutfactor(c::SizeConstraint) = c.constraint
            NaiveNASlib.minΔninfactor(c::SizeConstraint) = c.constraint
            # Can't have kwarg due to https://github.com/JuliaLang/julia/issues/32350
             av(in, outsize, constr, name="avs") = av(in, outsize, name=name, comp = SizeConstraint(constr))
             imu(in, outsize, name="imu") = MutationVertex(CompVertex(identity, in), IoSize(nout(in), outsize), NamedTrait(Immutable(), name))

            @testset "Add to nout-constrained stacking" begin
                v0 = inpt(3, "v0")
                v1 = av(v0, 8, 2, "v1")
                v2 = av(v0, 6, 3, "v2")
                v3 = sv(v1, name="v3")
                v4 = av(v3, 5, 5, "v4")
                v5 = av(v3, 7, 7, "v5")

                @test inputs(v3) == [v1]
                @test minΔninfactor(v3) == 70

                create_edge!(v2, v3)
                @test inputs(v3) == [v1, v2]

                @test nin(v3) == [nout(v1), nout(v2)] == [8, 105]
                @test [nout(v3)] == nin(v4) == nin(v5) == [113]
            end

            @testset "Add immutable to nout-constrained stacking" begin
                v0 = inpt(3, "v0")
                v1 = av(v0, 8, 2, "v1")
                v2 = sv(v1, name="v2")
                v3 = av(v2, 5, 5, "v3")

                @test inputs(v2) == [v1]
                @test minΔninfactor(v2) == 10

                create_edge!(v0, v2)
                @test inputs(v2) == [v1, v0]

                @test nin(v2) == [nout(v1), nout(v0)] == [10, 3]
                @test [nout(v2)] == nin(v3) == [13]
            end

            @testset "Add nout-constrained to stacking with one immutable output" begin
                v0 = inpt(3, "v0")
                v1 = av(v0, 8, 3, "v1")
                v2 = av(v0, 10, 2, "v2")
                v3 = sv(v1, name="v3")
                v4 = av(v3, 5, 5, "v4")
                v5 = imu(v3, 3, "v5")

                @test inputs(v3) == [v1]
                @test ismissing(minΔnoutfactor(v3))

                create_edge!(v2, v3)
                @test inputs(v3) == [v1, v2]

                @test nin(v3) == [nout(v1), nout(v2)] == [2, 6]
                @test [nout(v3)] == nin(v4) == nin(v5) == [8]
            end

            @testset "Add to nout-constrained invariant" begin
                v0 = inpt(3, "v0")
                v1 = av(v0, 8, 2, "v1")
                v2 = av(v0, 6, 3, "v2")
                v3 = iv(v1, name="v3")
                v4 = av(v3, 5, 5, "v4")
                v5 = av(v3, 7, 7, "v5")

                @test inputs(v3) == [v1]
                @test minΔninfactor(v3) == 70

                create_edge!(v2, v3)
                @test inputs(v3) == [v1, v2]

                @test nin(v3) == [nout(v1), nout(v2)] == [78, 78]
                @test [nout(v3)] == nin(v4) == nin(v5) == [78]
            end

        end
    end

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

        @testset "Remove one of many inputs to stacking" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 4, name="v1")
            v2 = av(v0, 5, name="v2")
            v3 = av(v0, 6, name="v3")
            v4 = sv(v1,v2,v3, name="v4")
            v5 = av(v4, 7, name="v5")

            # Note, input to v1 can not be changed, we must decrease
            # nin of v4 (and v5)
            remove!(v2)
            @test inputs(v4) == [v1, v0, v3]
            @test nin(v4) == [nout(v1), nout(v0), nout(v3)] == [4,3,6]
            @test nin(v5) == [nout(v4)] == [3+4+6]

            #Now lets try without connecting the inputs to v4
            remove!(v1, RemoveStrategy(ConnectNone(), ChangeNinOfOutputs((-nout(v1), missing, missing))))
            @test inputs(v4) == [v0, v3]
            @test nin(v4) == [nout(v0), nout(v3)] == [3, 6]
            @test nin(v5) == [nout(v4)] == [3+6]
        end

        @testset "Remove one of many inputs to stacking increase size" begin
            v0 = inpt(7, "v0")
            v1 = av(v0, 4, name="v1")
            v2 = av(v0, 5, name="v2")
            v3 = av(v0, 6, name="v3")
            v4 = sv(v1,v2,v3, name="v4")
            v5 = av(v4, 7, name="v5")

            # Note, input to v1 can not be changed, we must decrease
            # nin of v4 (and v5)
            remove!(v2)
            @test inputs(v4) == [v1, v0, v3]
            @test nin(v4) == [nout(v1), nout(v0), nout(v3)] == [4,7,6]
            @test nin(v5) == [nout(v4)] == [4+7+6]

            #Now lets try without connecting the inputs to v4
            remove!(v1, RemoveStrategy(ConnectNone(), ChangeNinOfOutputs((-nout(v1), missing, missing))))
            @test inputs(v4) == [v0, v3]
            @test nin(v4) == [nout(v0), nout(v3)] == [7, 6]
            @test nin(v5) == [nout(v4)] == [7+6]
        end

        @testset "Remove one of many inputs to invariant" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 5, name="v1")
            v2 = av(v0, 5, name="v2")
            v3 = av(v0, 5, name="v3")
            v4 = iv(v1,v2,v3, name="v4")
            v5 = av(v4, 7, name="v5")

            # Note, input to v1 can not be changed, we must decrease
            # nin of v4 (and v5)
            remove!(v2)
            @test inputs(v4) == [v1, v0, v3]
            @test nin(v4) == [nout(v1), nout(v0), nout(v3)] == [3,3,3]
            @test nin(v5) == [nout(v4)] == [3]

            #Now lets try without connecting the inputs to v4
            # NoSizeChange is just to avoid touching the input vertex
            remove!(v1, RemoveStrategy(ConnectNone(), NoSizeChange()))
            @test inputs(v4) == [v0, v3]
            @test nin(v4) == [nout(v0), nout(v3)] == [3, 3]
            @test nin(v5) == [nout(v4)] == [3]
        end

        @testset "Remove one of many inputs to invariant increase size" begin
            v0 = inpt(7, "v0")
            v1 = av(v0, 5, name="v1")
            v2 = av(v0, 5, name="v2")
            v3 = av(v0, 5, name="v3")
            v4 = iv(v1,v2,v3, name="v4")
            v5 = av(v4, 7, name="v5")

            # Note, input to v1 can not be changed, we must decrease
            # nin of v4 (and v5)
            remove!(v2)
            @test inputs(v4) == [v1, v0, v3]
            @test nin(v4) == [nout(v1), nout(v0), nout(v3)] == [7,7,7]
            @test nin(v5) == [nout(v4)] == [7]

            #Now lets try without connecting the inputs to v4
            # NoSizeChange is just to avoid touching the input vertex
            remove!(v1, RemoveStrategy(ConnectNone(), NoSizeChange()))
            @test inputs(v4) == [v0, v3]
            @test nin(v4) == [nout(v0), nout(v3)] == [7, 7]
            @test nin(v5) == [nout(v4)] == [7]
        end

        @testset "Remove input duplicated stacking" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 4, name="v1")
            v2 = av(v0, 5, name="v2")
            v3 = sv(v1,v2,v2,v1, name="v3")
            v4 = av(v3, 7, name="v4")

            remove!(v1)
            @test inputs(v3) == [v0, v2, v2, v0]
            @test nin(v3) == [nout(v0), nout(v2), nout(v2), nout(v0)] == [3,5,5,3]
            @test nin(v4) == [nout(v3)] == [3+5+5+3]

            #Now lets try without connecting the inputs to v3
            remove!(v2, RemoveStrategy(ConnectNone(), ChangeNinOfOutputs((missing, -nout(v2), missing, missing))))
            @test inputs(v3) == [v0, v0]
            @test nin(v3) == [nout(v0), nout(v0)] == [3,3]
            @test nin(v4) == [nout(v3)] == [3+3]
        end

        @testset "Remove input duplicated invariant" begin
            v0 = inpt(3, "v0")
            v1 = av(v0, 5, name="v1")
            v2 = av(v0, 5, name="v2")
            v3 = iv(v1,v2,v2,v1, name="v3")
            v4 = av(v3, 7, name="v4")

            remove!(v1)
            @test inputs(v3) == [v0, v2, v2, v0]
            @test nin(v3) == [nout(v0), nout(v2), nout(v2), nout(v0)] == [3,3,3,3]
            @test nin(v4) == [nout(v3)] == [3]

            #Now lets try without connecting the inputs to v3
            # NoSizeChange is just to avoid touching the input vertex
            remove!(v2, RemoveStrategy(ConnectNone(), NoSizeChange()))
            @test inputs(v3) == [v0, v0]
            @test nin(v3) == [nout(v0), nout(v0)] == [3,3]
            @test nin(v4) == [nout(v3)] == [3]
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
