
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

    @testset "SelectDirection" begin

        mutable struct TestProbe <: AbstractSelectionStrategy
            vs
            function TestProbe(v)
                tp = new(nothing)
                Δoutputs(SelectDirection(tp), v, v -> error("Shall not be called!"))
                return tp
            end
        end
        NaiveNASlib.Δoutputs(s::TestProbe, vs::AbstractVector{<:AbstractVertex}, vfun::Function) = s.vs = Set(vs)

        v1 = av(iv(3), 5, "v1")
        v2 = av(v1, 4, "v2")
        v3 = av(v2, 3, "v3")

        tp = TestProbe(v2)
        @test tp.vs == nothing

        Δnin(v2, -1)
        tp = TestProbe(v2)
        @test tp.vs == Set([v2, v1])

        Δnout(v2, 1)
        tp = TestProbe(v2)
        @test tp.vs == Set([v1, v2, v3])

        Δnin(v2, 1)
        tp = TestProbe(v2)
        @test tp.vs == Set([v2, v3])

    end

    @testset "ApplyAfter" begin
        v1 = av(iv(3), 5, "v1")
        v2 = av(v1, 4, "v2")
        v3 = av(v2, 3, "v3")

        res = []
        apply(v) = push!(res, v)

        Δnout(v2, -1)
        @test Δoutputs(ApplyAfter(apply, OutSelectExact()), v2, v -> 1:nout_org(v))

        @test res == [v2, v3]
    end

    @testset "Absorb 2 Absorb" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        g = CompGraph(inpt, v2)

        Δnout(v1, -2)
        @test Δoutputs(v1, v -> 1:nout_org(v))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [3,4,5]
        apply_mutation(g)

        @test size(g(ones(1, 3))) == (1, nout(v2))

        Δnout(v1, 3)
        @test Δoutputs(Output(), v1, v->1:nout_org(v))

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
        @test !Δoutputs(NoutRevert(), v1, v->1:nout_org(v))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [1,2,3,4,5]
        apply_mutation(g)

        @test size(g(ones(1, 3))) == (1, nout(v2))

        Δnout(v1, +3)

        @test !Δoutputs(NoutRevert(), v1, v-> 1:nout_org(v))

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [1,2,3,4,5]
        apply_mutation(g)

        @test size(g(ones(1, 3))) == (1, nout(v2))
    end

    @testset "Absorb 2 Absorb fail" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        Δnout(v1, -2)
        @test_throws ErrorException Δoutputs(SelectionFail(), v1, v -> 1:nout_org(v))
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

        @test Δoutputs(v4, v -> 1:nout_org(v))
        apply_mutation(g)

        @test nout(v1) == 5
        @test nout(v2) == 3

        @test size(g(ones(1, 3))) == (1, nout(v4))

        Δnout(v4, 6)

        @test nout(v1) == 7
        @test nout(v2) == 5

        @test Δoutputs(Output(), v4, v->1:nout_org(v))
        apply_mutation(g)

        @test nout(v1) == 7
        @test nout(v2) == 5

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

        @test Δoutputs(v6, v->1:nout_org(v))
        apply_mutation(g)

        @test nout(v1) == 5
        @test nout(v2) == 2
        @test nout(v3) == 3

        @test size(g(ones(1, 3))) == (1, nout(v6))

        Δnout(v6, 6)

        @test nout(v1) == 9
        @test nout(v2) == 4
        @test nout(v3) == 5

        @test Δoutputs(Output(), v6, v->1:nout_org(v))
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
        @test Δoutputs(OutSelect{NaiveNASlib.Relaxed}(SelectionFail()), v2, v -> v == v2 ? (-nout(inpt):nout_org(v1)-1) : 1:nout_org(v))
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
        Δnout(v7, -6)

        @test nout(v1) == 7
        @test nout(v2) == 4
        @test nout(v3) == 8
        @test nout(v4) == 4

        @test @test_logs (:warn, "Selection for vertex v7 failed! Relaxing size constraint...")  match_mode=:any Δoutputs(v7, v->-1:nout_org(v)-2)
        apply_mutation(g)

        @test nout(v1) == 6
        @test nout(v2) == 3
        @test nout(v3) == 8
        @test nout(v4) == 5

        @test size(g(ones(1, 3))) == (1, nout(v7))

        Δnout(v7, 4)

        @test nout(v1) == 8
        @test nout(v2) == 4
        @test nout(v3) == 9
        @test nout(v4) == 5

        # Works on the first try this time around
        @test Δoutputs(Output(), v7, v->1:nout_org(v))
        apply_mutation(g)

        @test nout(v1) == 8
        @test nout(v2) == 4
        @test nout(v3) == 9
        @test nout(v4) == 5

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
        Δnout(v7, 6)

        @test nout(v1) == 4
        @test nout(v2) == 6
        @test nout(v3) == 6
        @test nout(v4) == 8

        @test @test_logs (:warn, "Selection for vertex v7 failed! Relaxing size constraint...")  match_mode=:any Δoutputs(Output(), v7, v -> 1:nout_org(v))
        apply_mutation(g)

        # Sizes can't change when increasing, even if problem is relaxed :(
        @test nout(v1) == 4
        @test nout(v2) == 6
        @test nout(v3) == 6
        @test nout(v4) == 8

        @test size(g(ones(1, 3))) == (1, nout(v7))
    end

    @testset "SizeStack increase decrease" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 10, "v2")
        v3 = cc(v1,v2, name="v3")
        v4 = av(v3, 4, "v4")

        g = CompGraph(inpt, v3)
        @test size(g(ones(1, 3))) == (1, nout(v3))

        Δnout(v3, -5)
        Δnout(v1, 3)

        @test nin(v3) == [nout(v1), nout(v2)] == [5, 6]
        @test nout(v3) == sum(nin(v3)) == 11

        @test Δoutputs(Output(), v3, v -> 1:nout_org(v))

        @test in_inds(op(v3)) == [out_inds(op(v1)), out_inds(op(v2))] == [[1,2,3,-1,-1],[5,6,7,8,9,10]]
        @test [out_inds(op(v3))] == in_inds(op(v4)) == [[1,2,3,-1,-1,8,9,10,11,12,13]]

        apply_mutation(g)
        @test size(g(ones(1, 3))) == (1, nout(v3))

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

        Δnout(v2, -1)

        @test nout(v6) == 4
        @test nout(v7) == 13
        @test nout(v0) == 9

        @test Δoutputs(v1, v -> 1:nout_org(v))
        apply_mutation(g)

        @test nout(v6) == 4
        @test nout(v7) == 13
        @test nout(v0) == 9

        @test size.(g(ones(1,3))) == ((1, nout(v1)), (1, nout(v9)))

        Δnout(v2, 5)

        @test nout(v6) == 4
        @test nout(v7) == 16
        @test nout(v0) == 14

        @test @test_logs (:warn, "Selection for vertex v1 failed! Relaxing size constraint...") Δoutputs(Output(), v1, v->1:nout_org(v))
        apply_mutation(g)

        @test nout(v6) == 4
        @test nout(v7) == 16
        @test nout(v0) == 14

        @test size.(g(ones(1,3))) == ((1, nout(v1)), (1, nout(v9)))
    end

    @testset "Increase after vertex removal" begin
        inpt = iv(3)
        v1 = av(inpt, 2, "v1")
        v2 = tv(v1, "v2")
        v3 = av(v2, 4, "v3")
        v4 = av(v3, 3, "v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))

        remove!(v3, RemoveStrategy())

        @test Δoutputs(v2, v -> 1:nout_org(v))

        @test in_inds(op(v4))[] == [1,2,-1,-1] # TODO: Should be [1,2,3,4]
        @test out_inds(op(v1)) == out_inds(op(v2)) == [1, 2, -1, -1]

        apply_mutation(g)

        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))
    end

    @testset "Decrease after vertex removal" begin
        inpt = iv(3)
        v1 = av(inpt, 2, "v1")
        v2 = tv(v1, "v2")
        v3 = av(v2, 4, "v3")
        v4 = av(v3, 3, "v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))

        remove!(v3, RemoveStrategy(DecreaseBigger()))

        # What happened now is that nin(v4) got decreased from 4 to 2. We now need to select which inputs to keep
        # However, there is absolutely no need at all to select anything from v2 and before as they have not changed.
        @test Δoutputs(Output(), v2, v -> 1:nout_org(v))

        @test in_inds(op(v4))[] == out_inds(op(v1)) == out_inds(op(v2)) == out_inds(op(v3)) == [1, 2]

        apply_mutation(g)

        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))
    end

    @testset "Increase after vertex removal SizeStack" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(v1, 5, "v3")
        v4 = cc(v2, v3, name="v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))

        remove!(v3, RemoveStrategy())

        @test Δoutputs(Output(), v2, v -> 1:nout_org(v))

        @test in_inds(op(v4)) == [out_inds(op(v2)), out_inds(op(v1))] == [[1,2,3,4], [1,2,3,-1,-1]]
        @test out_inds(op(v4)) == 1:9

        apply_mutation(g)

        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))
    end

    @testset "Decrease after vertex removal SizeStack" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(v1, 5, "v3")
        v4 = cc(v2, v3, name="v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))

        remove!(v3, RemoveStrategy(DecreaseBigger()))

        @test Δoutputs(v2, v -> 1:nout_org(v))

        @test in_inds(op(v4)) == [out_inds(op(v2)), out_inds(op(v1))] == [[1,2,3,4], [1,2,3]]
        @test out_inds(op(v4)) == [1,2,3,4,7,8,9]

        apply_mutation(g)

        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))
    end

    @testset "SelectOutputs after increse due to vertex removal" begin
        inpt = iv(3)
        v1 = av(inpt, 2, "v1")
        v2 = av(v1, 3, "v2")
        v3 = av(v2, 5, "v3")
        v4 = av(v3, 3, "v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))

        remove!(v3, RemoveStrategy(SelectOutputs(select=SelectDirection(), valuefun = v -> 1:nout_org(v))))

        @test out_inds(op(v2)) == [1,2,3,-1,-1]
        @test in_inds(op(v4))[] == [1,2,3,4,5]

        apply_mutation(g)

        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))
    end

    @testset "SelectOutputs after decrease due to vertex removal" begin
        inpt = iv(3)
        v1 = av(inpt, 2, "v1")
        v2 = av(v1, 3, "v2")
        v3 = av(v2, 5, "v3")
        v4 = av(v3, 3, "v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))

        remove!(v3, RemoveStrategy(SelectOutputs(align=DecreaseBigger(), valuefun= v -> 1:nout_org(v))))

        @test in_inds(op(v4))[] == out_inds(op(v3)) == [3,4,5]

        apply_mutation(g)

        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))
    end

    @testset "ApplyMutation vertex removal SizeStack" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(v1, 5, "v3")
        v4 = cc(v2, v3, name="v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))

        remove!(v3, RemoveStrategy(ApplyMutation()))
        @test nout(v1) == 5

        @test size(g(ones(Float32, 1,3))) == (1, nout(v4))
    end

    @testset "TruncateInIndsToValid" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = tv(v1, "v2")
        v3 = av(v2, 2, "v3")

        g = CompGraph(inpt, v2)
        @test size(g(ones(Float32, 1,3))) == (1, nout(v2))

        vnew = av(inpt, 7, "vnew")
        create_edge!(vnew, v2;strategy=PostAlignJuMP())
        @test nin_org(v2) == [3, 0] != [nout(v1), nout(vnew)]

        Δoutputs(TruncateInIndsToValid(), g, v -> 1:nout_org(v))

        @test out_inds(op(vnew)) == [4,5,6,7]
        @test in_inds(op(v2)) == [[1,2,3,-1], [1,2,3,-1]]
        @test out_inds(op(v2)) == [1,2,3,-1]
    end

    @testset "CompConstraint" begin

        struct CompConstraint end
        function NaiveNASlib.compconstraint!(::AbstractJuMPSelectionStrategy, ::CompConstraint, data)
            var = data.outselectvars[data.vertex];
            JuMP.@constraint(data.model, var[[1, 3]] .== 0)
        end

        inpt = iv(3)
        v1 = absorbvertex(CompConstraint(), 4, inpt, traitdecoration = tf("v1"))
        v2 = av(v1, 5, "v2")

        Δnout(v1, 3)
        @test @test_logs (:warn, "Selection for vertex v1 failed! Relaxing size constraint...") Δoutputs(Output(), v1, v->nout_org(v):-1:1)

        @test out_inds(op(v1)) == in_inds(op(v2))[] == [2, 4, -1, -1, -1, -1, -1]
    end
end
