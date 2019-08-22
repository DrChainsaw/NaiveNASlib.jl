
@testset "Selection" begin

    # Helper methods
    nt(name) = t -> NamedTrait(t, name)
    tf(name) = t -> nt(name)(t)
    iv(size, name="in") = inputvertex(name, size)
    av(in, outsize, name) = absorbvertex(MatMul(nout(in), outsize), outsize, in, traitdecoration=tf(name))
    tv(in, name) = invariantvertex(identity, in, traitdecoration=tf(name))

    cc(ins...; name) = conc(ins...; dims=2, traitdecoration=tf(name))
    nc(name) = traitconf(nt(name))

    select_outputs_and_change(v, values) = select_outputs_and_change(NoutExact(), v, values)
    function select_outputs_and_change(s, v, values)
        execute, selected = select_outputs(s, v, values)
        if execute
            Δnout(v, selected)
        end
    end

    @testset "Absorb 2 Absorb" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        g = CompGraph(inpt, v2)

        Δnout(v1, -2)
        select_outputs_and_change(v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [3,4,5]
        apply_mutation(g)

        @test size(g(ones(1, 3))) == (1, nout(v2))

        Δnout(v1, 3)
        select_outputs_and_change(v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [1,2,3,-1,-1,-1]
        apply_mutation(g)

        @test size(g(ones(1, 3))) == (1, nout(v2))
    end

    @testset "Absorb 2 Absorb revert" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        g = CompGraph(inpt, v2)

        Δnout(v1, -2)
        select_outputs_and_change(NoutRevert(), v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [1,2,3,4,5]
        apply_mutation(g)

        @test size(g(ones(1, 3))) == (1, nout(v2))

        Δnout(v1, +3)

        select_outputs_and_change(NoutRevert(), v1, 1:nout_org(op(v1)))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [1,2,3,4,5]
        apply_mutation(g)

        @test size(g(ones(1, 3))) == (1, nout(v2))
    end

    @testset "Absorb 2 Absorb fail" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        Δnout(v1, -2)
        @test_throws ErrorException select_outputs_and_change(SelectionFail(), v1, 1:nout_org(op(v1)))
    end

    @testset "SizeStack duplicate" begin
        inpt = iv(3)
        v1 = av(inpt, 7, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = cc(v2, v1, name="v3")
        v4 = cc(v3, v2, name="v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(1, 3))) == (1, nout(v4))

        @test minΔnoutfactor(v4) == 2
        Δnout(v4, -4)

        @test nout(v1) == 5
        @test nout(v2) == 3

        select_outputs_and_change(v4, 1:nout_org(op(v4)))
        apply_mutation(g)

        @test nout(v1) == 5
        @test nout(v2) == 3

        @test size(g(ones(1, 3))) == (1, nout(v4))

        Δnout(v4, 6)

        @test nout(v1) == 9
        @test nout(v2) == 4

        select_outputs_and_change(v4, 1:nout_org(op(v4)))
        apply_mutation(g)

        @test nout(v1) == 9
        @test nout(v2) == 4

        @test size(g(ones(1, 3))) == (1, nout(v4))
    end

    @testset "SizeInvariant duplicate" begin
        inpt = iv(3)
        v1 = av(inpt, 7, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(inpt, 3, "v3")

        v4 = cc(v1, v2, name="v4")
        v5 = cc(v2, v3, v2, name="v5")

        v6 = nc("v6") >> v4 + v5

        g = CompGraph(inpt, v6)
        @test size(g(ones(1, 3))) == (1, nout(v6))

        @test minΔnoutfactor(v6) == 2
        Δnout(v6, -4)

        @test nout(v1) == 5
        @test nout(v2) == 2
        @test nout(v3) == 3

        select_outputs_and_change(v6, 1:nout_org(op(v6)))
        apply_mutation(g)

        @test nout(v1) == 5
        @test nout(v2) == 2
        @test nout(v3) == 3

        @test size(g(ones(1, 3))) == (1, nout(v6))

        Δnout(v6, 6)

        @test nout(v1) == 9
        @test nout(v2) == 4
        @test nout(v3) == 5

        select_outputs_and_change(v6, 1:nout_org(op(v6)))
        apply_mutation(g)

        @test nout(v1) == 9
        @test nout(v2) == 4
        @test nout(v3) == 5

        @test size(g(ones(1, 3))) == (1, nout(v6))
    end

    @testset "SizeStack one immutable" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = cc(inpt, v1, name="v2")

        g = CompGraph(inpt, v2)
        @test size(g(ones(1, 3))) == (1, nout(v2))

        Δnout(v1, -3)

        @test nin(v2) == [nout(inpt), nout(v1)] == [3, 2]
        @test nout(v2) == 5

        # "Tempt" optimizer to not select inputs from inpt
        select_outputs_and_change(NoutRelaxSize(0.5, 1), v2, -nout(inpt):nout_org(op(v1))-1)
        apply_mutation(g)

        @test nin(v2) == [nout(inpt), nout(v1)] == [3, 2]
        @test nout(v2) == 5

        @test size(g(ones(1, 3))) == (1, nout(v2))
    end

    @testset "SizeInvariant exact infeasible" begin
        inpt = iv(3)
        v1 = av(inpt, 10, "v1")
        v2 = av(inpt, 5, "v2")
        v3 = av(inpt, 10, "v3")
        v4 = av(inpt, 5, "v4")

        v5 = cc(v1, v2, v3, name="v5")
        v6 = cc(v2, v1, v2, v4, name="v6")

        v7 = nc("v7") >> v5 + v6

        g = CompGraph(inpt, v7)
        @test size(g(ones(1, 3))) == (1, nout(v7))

        @test minΔnoutfactor(v7) == 2
        Δnout(v7, -4)

        @test nout(v1) == 8
        @test nout(v2) == 5
        @test nout(v3) == 8
        @test nout(v4) == 3

        @test_logs (:warn, "Selection for vertex v7 failed! Relaxing size constraint...")  match_mode=:any select_outputs_and_change(v7, 1:nout_org(op(v7)))
        apply_mutation(g)

        @test nout(v1) == 6
        @test nout(v2) == 3
        @test nout(v3) == 6
        @test nout(v4) == 3

        @test size(g(ones(1, 3))) == (1, nout(v7))

        Δnout(v7, 20)

        @test nout(v1) == 14
        @test nout(v2) == 7
        @test nout(v3) == 14
        @test nout(v4) == 7

        # Works on the first try this time around
        select_outputs_and_change(v7, 1:nout_org(op(v7)))
        apply_mutation(g)

        @test nout(v1) == 14
        @test nout(v2) == 7
        @test nout(v3) == 14
        @test nout(v4) == 7

        @test size(g(ones(1, 3))) == (1, nout(v7))
    end

    @testset "SizeInvariant increase exact infeasible" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(inpt, 4, "v3")
        v4 = av(inpt, 5, "v4")

        v5 = cc(v1, v2, v4, name="v5")
        v6 = cc(v3, v4, v1, name="v6")

        v7 = nc("v7") >> v5 + v6

        g = CompGraph(inpt, v7)
        @test size(g(ones(1, 3))) == (1, nout(v7))

        @test minΔnoutfactor(v7) == 1
        Δnout(v7, 5)

        @test nout(v1) == 4
        @test nout(v2) == 5
        @test nout(v3) == 5
        @test nout(v4) == 8

        @test_logs (:warn, "Selection for vertex v7 failed! Relaxing size constraint...")  match_mode=:any select_outputs_and_change(v7, 1:nout_org(op(v7)))
        apply_mutation(g)

        # Sizes can't change when increasing, even if problem is relaxed :(
        @test nout(v1) == 4
        @test nout(v2) == 5
        @test nout(v3) == 5
        @test nout(v4) == 8

        @test size(g(ones(1, 3))) == (1, nout(v7))
    end

    @testset "Constrained by remote subtree" begin
        inpt = iv(3)
        v0 = av(inpt, 10, "v0")

        # Main branch, the one we want to change
        v1 = cc(v0, v0, name="v1")

        # Path to subtree
        # First hop
        v2 = av(inpt, 10, "v2")
        v3 = nc("v3") >> v2 + v0

        # Second hop (v4a and v4b should not be touhed)
        v4a = av(inpt, 3, "v4a")
        v4b = av(inpt, 5, "v4b")
        v5 = cc(v4a, v3, v4b, name="v5")

        # Subtree
        v6 = av(inpt, 4, "v6")
        v7 = av(inpt, nout(v5) - nout(v6), "v7")
        v8 = cc(v6,v7,name="v8")

        # Aaaand connect it to the path
        v9 = nc("v9") >> v8 + v5
        v10 = av(v9, 5, "v10")

        g = CompGraph(inpt, [v1, v9])
        @test size.(g(ones(1,3))) == ((1, nout(v1)), (1, nout(v9)))

        @test minΔnoutfactor(v1) == 2

        Δnout(v2, -6)

        @test nout(v6) == 3
        @test nout(v7) == 9
        @test nout(v0) == 4


        @test_logs (:warn, "Selection for vertex v1 failed! Relaxing size constraint...")  match_mode=:any select_outputs_and_change(v1, 1:nout_org(v1))
        apply_mutation(g)

        @test nout(v6) == 4
        @test nout(v7) == 10
        @test nout(v0) == 6

        @test size.(g(ones(1,3))) == ((1, nout(v1)), (1, nout(v9)))

        Δnout(v2, 8)

        @test nout(v6) == 6
        @test nout(v7) == 16
        @test nout(v0) == 14

        select_outputs_and_change(v1, 1:nout_org(v1))
        apply_mutation(g)

        @test nout(v6) == 6
        @test nout(v7) == 16
        @test nout(v0) == 14

        @test size.(g(ones(1,3))) == ((1, nout(v1)), (1, nout(v9)))
    end

    @testset "NoutMainVar after vertex removal" begin
        inpt = iv(3)
        v1 = av(inpt, 2, "v1a")
        v2 = tv(v1, "v2")
        v3 = av(v2, 4, "v3")
        v4 = av(v3, 3, "v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))

        remove!(v3, RemoveStrategy(DecreaseBigger()))

        # What happened now is that nin(v4) got decreased from 4 to 2. We now need to select which inputs to keep
        # However, there is absolutely no need at all to select anything from v2 and before as they have not changed.

        # Approach used: Select best nout(v2) outputs from v3 (hoping that the best outputs for v3 are also the best inputs for v4)

        # out=false because we are actually selecting for v4 in the input direction
        cdict = validouts(v2, Set([v2]), Set(AbstractVertex[]), false)
        valid, selinds = select_outputs(NoutMainVar(NoutExact(), NoutExact()), v2, 1:nout_org(v3), cdict)

        # Don't want to propagate to v2!
        s = NaiveNASlib.VisitState{Vector{Int}}(v2)
        NaiveNASlib.visited_out!.(s, [v2])
        Δnin(v4, selinds, s=s)

        @test in_inds(op(v4))[] == [3, 4]
        @test out_inds(op(v1)) == out_inds(op(v2)) == out_inds(op(v3)) == out_inds(op(v2)) == [1, 2]

        apply_mutation(g)

        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))
    end

    @testset "NoutMainVar exact infeasible" begin
        inpt = iv(3)
        v1 = av(inpt, 2, "v1")
        v2 = av(inpt, 2, "v2")
        v3 = cc(v1, v2, v1, v2, name="v3")
        v4 = av(v3, 3, "v4")

        g = CompGraph(inpt, v3)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v3))

        Δnout(v3, 6)

        @test nout(v3) == 2nout(v1) + 2nout(v2) == 14
        @test nout(v1) == 3
        @test nout(v2) == 4

        @test_logs (:warn, "Selection for vertex v3 failed! Relaxing size constraint...") select_outputs_and_change(NoutMainVar(NoutExact(), NoutRelaxSize()), v3, 1:(nout_org(v3)-1))

        @test nout(v3) == 2nout(v1) + 2nout(v2) == 14
        @test nout(v1) == 3
        @test nout(v2) == 4

        @test in_inds(op(v4))[] == out_inds(op(v3)) == [1, 2, -1, 3, -1, -1, -1, 5, 6, -1, 7, -1, -1, -1]
        @test out_inds(op(v1)) ==  [1, 2, -1]
        @test out_inds(op(v2)) ==  [1, -1, -1, -1]

        apply_mutation(g)

        @test size(g(ones(Float32, 1,3))) == (1, nout(v3))
    end
end