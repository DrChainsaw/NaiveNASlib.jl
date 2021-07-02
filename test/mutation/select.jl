
@testset "Selection" begin

    # Helper methods
    nt(name) = t -> NamedTrait(t, name)
    tf(name) = t -> nt(name)(t)
    iv(size, name="in") = inputvertex(name, size)
    av(in, outsize, name) = absorbvertex(IndMem(MatMul(nout(in), outsize)), in; traitdecoration=tf(name))
    tv(in, name) = invariantvertex(IndMem(identity, [nout(in)], nout(in)), in; traitdecoration=tf(name))

    concindmem(ins) = f ->  IndMem(f, [nout(v) for v in ins], sum(nout, ins))
    cc(ins...; name) = conc(ins...; dims=1, traitdecoration=tf(name), outwrap=concindmem(ins))
    nc(name) = traitconf(nt(name))

    @testset "Split relaxed exact" begin
        import NaiveNASlib: split_exact_relaxed

        @test (:a => relaxed(3)) == (:a => 3 => Relaxed())

        @test split_exact_relaxed((:a=>2, :b=>3=>Relaxed(), :c=>4=>Exact()),) == (Dict(:a => 2, :c => 4), Dict(:b => 3))
        @test split_exact_relaxed((Dict(:a=>2, :b=>3=>Relaxed(), :c=>4=>Exact())),) == (Dict(:a => 2, :c => 4), Dict(:b => 3))
    end

    @testset "Make ΔNout" begin
        import NaiveNASlib: Exact, Relaxed
        @testset "All Exact" begin
            res = ΔNout(:a => 2, :b => 3 => Exact())
            @test res isa ΔNout{Exact} 
            @test res.Δs == Dict(:a => 2, :b => 3)           
        end

        @testset "All Relaxed" begin
            res = ΔNout(:a => 2 => Relaxed(), :b => relaxed(3))
            @test res isa ΔNout{Relaxed} 
            @test res.Δs == Dict(:a => 2, :b => 3)           
        end

        @testset "Mix" begin
            import NaiveNASlib: ΔNoutMix
            res = ΔNout(:a => 2, :b => relaxed(3), :c => 4, :d => relaxed(5))
            @test res isa ΔNoutMix
            @test res.exact.Δs == Dict(:a => 2, :c => 4)
            @test res.relax.Δs == Dict(:b => 3, :d => 5)
        end
    end

    @testset "Join extraced inds" begin
        import NaiveNASlib: join_extracted_inds     

        @test join_extracted_inds([2,3,4,5], [0,0,0,0]) == [2,3,4,5]
        @test join_extracted_inds([2,3,4,5], [1,0,0,0,0,0,1]) == [2,-1,3,4,5,-1]
        @test join_extracted_inds([2,3,4,5], [1,2,3,4]) == [2,-1,3,-1,-1,4,-1,-1,-1,5,-1,-1,-1,-1,]
        @test join_extracted_inds([2], [2,4,0]) == [2,-1,-1,-1,-1,-1,-1]
        @test join_extracted_inds([], [3,0]) == [-1,-1,-1]
    end

    @testset "SelectDirection" begin

        mutable struct TestProbe <: AbstractΔSizeStrategy
            vs
            d
            function TestProbe(v, d)
                tp = new(nothing, d)
                Δsize(v -> error("Shall not be called!"), SelectDirection(tp), v)
                return tp
            end
        end
        NaiveNASlib.Δsize(vfun, s::TestProbe, d, v::AbstractVertex) = s.vs = Set(all_in_Δsize_graph(v, d))
        NaiveNASlib.Δdirection(t::TestProbe) = t.d

        v1 = av(iv(3), 5, "v1")
        v2 = av(v1, 4, "v2")
        v3 = av(v2, 3, "v3")

        tp = TestProbe(v2, nothing)
        @test tp.vs === nothing

        tp = TestProbe(v2, Input())
        @test tp.vs == Set([v2, v1])

        tp = TestProbe(v2, Output())
        @test tp.vs == Set([v2, v3])

        tp = TestProbe(v2, Both())
        @test tp.vs == Set([v1, v2, v3])
    end

    @testset "Argument plumbing" begin
        function graphgen() 
            inpt = iv(3)
            v0 = av(inpt, 3, "v0")
            v1 = av(v0, 5, "v1")
            v2 = av(v1, 4, "v2")
            return v1, v2
        end

        @testset "Δnout$(use_fun ? " with value function" : "")" for use_fun in (false, true)
            f = use_fun ? (v -> (1:nout(v)),) : ()
            @testset "Single Δnout$(wrap === identity ? "" : " $wrap")" for wrap in (identity, relaxed)
                v1,v2 = graphgen()
                @test Δnout(f..., v1, wrap(3))
                @test nout.((v1, v2)) == (8, 4)
            end

            @testset "Single Δnout pair$(wrap === identity ? "" : " $wrap")" for wrap in (identity, relaxed)
                v1,v2 = graphgen()
                @test Δnout(f..., v1=>wrap(3))
                @test nout.((v1, v2)) == (8, 4)
            end

            @testset "Single Δnout Dict$(wrap === identity ? "" : " $wrap")" for wrap in (identity, relaxed)
                v1,v2 = graphgen()
                @test Δnout(f..., Dict(v1=>wrap(3)))
                @test nout.((v1, v2)) == (8, 4)
            end

            @testset "Multi Δnout pair$(wrap === identity ? "" : " $wrap")" for wrap in (identity, relaxed)
                v1,v2 = graphgen()
                @test Δnout(f..., v1=>wrap(3), v2=>wrap(2))
                @test nout.((v1, v2)) == (8, 6)
            end

            @testset "Multi Δnout Dict$(wrap === identity ? "" : " $wrap")" for wrap in (identity, relaxed)
                v1,v2 = graphgen()
                @test Δnout(f..., Dict(v1=>wrap(3), v2=>wrap(2)))
                @test nout.((v1, v2)) == (8, 6)
            end

            @testset "Multi Δnout pair mix" begin
                v1,v2 = graphgen()
                @test Δnout(f..., v1=>3, v2=>relaxed(2))
                @test nout.((v1, v2)) == (8, 6)
            end

            @testset "Multi Δnout Dict mix" begin
                v1,v2 = graphgen()
                @test Δnout(f..., Dict(v1=>3, v2=>relaxed(2)))
                @test nout.((v1, v2)) == (8, 6)
            end
        end

        @testset "Δnin$(use_fun ? " with value function" : "")" for use_fun in (false, true)
            f = use_fun ? (v -> (1:nout(v)),) : ()
            @testset "Single Δnin$(wrap === identity ? "" : " $wrap")" for wrap in (identity, relaxed)
                v1,v2 = graphgen()
                @test Δnin(f..., v1, wrap(3))
                @test nin.((v1, v2)) == ([6], [5])
            end

            @testset "Single Δnin pair$(wrap === identity ? "" : " $(repr(wrap))")" for wrap in 
            (identity, tuple, relaxed, relaxed ∘ tuple, tuple ∘ relaxed)
                v1,v2 = graphgen()
                @test Δnin(f..., v1=>wrap(3))
                @test nin.((v1, v2)) == ([6], [5])
            end

            @testset "Single Δnin Dict$(wrap === identity ? "" : " $(repr(wrap))")" for wrap in 
            (identity, tuple, relaxed, relaxed ∘ tuple, tuple ∘ relaxed)
                v1,v2 = graphgen()
                @test Δnin(f..., Dict(v1=>wrap(3)))
                @test nin.((v1, v2)) == ([6], [5])
            end

            @testset "Multi Δnin pair mix $(repr(wrap1)) and $(repr(wrap2)) " for wrap1 in 
            (identity, tuple, relaxed, relaxed ∘ tuple, tuple ∘ relaxed), wrap2 in 
            (identity, tuple, relaxed, relaxed ∘ tuple, tuple ∘ relaxed)
                v1,v2 = graphgen()
                @test Δnin(f..., v1=>wrap1(3), v2=>wrap2(2))
                @test nin.((v1, v2)) == ([6], [7])
            end

            @testset "Multi Δnin Dict $(repr(wrap1)) and $(repr(wrap2)) " for wrap1 in 
                (identity, tuple, relaxed, relaxed ∘ tuple, tuple ∘ relaxed), wrap2 in 
                (identity, tuple, relaxed, relaxed ∘ tuple, tuple ∘ relaxed)
                v1,v2 = graphgen()
                @test Δnin(f..., Dict(v1=>wrap1(3), v2=>wrap2(2)))
                @test nin.((v1, v2)) == ([6], [7])
            end
        end
    end

    @testset "Absorb 2 Absorb" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        g = CompGraph(inpt, v2)

        @test Δnout(v -> 1:nout(v), v1, -2)

        @test lastouts(v1) == lastins(v2) == [3,4,5]
        @test size(g(ones(3))) == (nout(v2),)

        @test Δnout(v1, 3)

        @test lastouts(v1) == lastins(v2) == [1,2,3,-1,-1,-1]
        @test size(g(ones(3))) == (nout(v2),)

        @test Δnin(v2, -2)
        @test lastouts(v1) == lastins(v2) == [3,4,5,6]
        @test size(g(ones(3))) == (nout(v2),)
    end

    @testset "Absorb 2 Absorb fail" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = av(v1, 4, "v2")

        @test_throws ErrorException Δsize(v -> 1:nout_org(v), ΔSizeFailError("Success!?"), v1)
    end

    @testset "SizeStack duplicate" begin
        inpt = iv(3)
        v1 = av(inpt, 7, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = cc(v2, v1, name="v3")
        v4 = cc(v3, v2, name="v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(3,2))) == (nout(v4), 2)

        @test minΔnoutfactor(v4) == 2
        @test Δnout(v4, -4)

        @test nout(v1) == 7
        @test nout(v2) == 2

        @test size(g(ones(3))) == (nout(v4),)

        @test Δnout(v4, 6)

        @test nout(v1) == 11
        @test nout(v2) == 3

        @test size(g(ones(3, 3))) == (nout(v4), 3)

        @test Δnin(v4, relaxed(-2), 3)

        @test nout(v1) == 6
        @test nout(v2) == 6

        @test size(g(ones(3))) == (nout(v4),)

        @test Δnin(v4, -2, missing)

        @test nout(v1) == 4
        @test nout(v2) == 6

        @test size(g(ones(3))) == (nout(v4),)
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
        @test size(g(ones(3))) == (nout(v6),)

        @test minΔnoutfactor(v6) == 2
        @test Δnout(v->1:nout(v), v6, -4)

        @test nout(v1) == 5
        @test nout(v2) == 2
        @test nout(v3) == 3

        @test size(g(ones(3))) == (nout(v6),)

        @test Δnout(v6, 6)

        @test nout(v1) == 9
        @test nout(v2) == 4
        @test nout(v3) == 5

        @test size(g(ones(3))) == (nout(v6),)

        @test Δnin(v -> 1:nout(v), v6, -2, relaxed(-2))

        @test nout(v1) == 8
        @test nout(v2) == 3
        @test nout(v3) == 5

        @test size(g(ones(3))) == (nout(v6),)

        @test Δnin(v -> 1:nout(v), v6, 3, missing)

        @test nout(v1) == 10
        @test nout(v2) == 4
        @test nout(v3) == 6

        @test size(g(ones(3))) == (nout(v6),)
    end

    @testset "SizeStack exact infeasible" begin
        inpt = iv(3)
        v1 = av(inpt, 4, "v1")
        v2a = tv(v1, "v2a")
        v2b = tv(v1, "v2b")
        v2c = tv(v1, "v2c")
        v3 = cc(v2a, v2b, v2c; name="v3")

        g = CompGraph(inpt, v3)
        @test size(g(ones(3))) == (nout(v3),)

        @test @test_logs (:warn, r"Could not change nout of") Δnout(v3, 2)

        @test nout(v3) == 3nout(v1) == 15

        @test size(g(ones(3))) == (nout(v3),)
    end

    @testset "SizeStack one immutable" begin
        inpt = iv(3)
        v1 = av(inpt, 5, "v1")
        v2 = cc(inpt, v1, name="v2")

        g = CompGraph(inpt, v2)
        @test size(g(ones(3))) == (nout(v2),)

        @test Δnout(v1, -3) do v
            # "Tempt" optimizer to not select inputs from inpt
            v === v2 && return (-nout(inpt):nout_org(v1)-1) 
            return 1:nout_org(v)
        end

        @test nin(v2) == [nout(inpt), nout(v1)] == [3, 2]
        @test nout(v2) == 5

        @test size(g(ones(3))) == (nout(v2),)
    end

    @testset "Decrease SizeStack and SizeInvariant" begin
        inpt = iv(3)
        v1 = av(inpt, 10, "v1")
        v2 = av(inpt, 5, "v2")
        v3 = av(inpt, 10, "v3")
        v4 = av(inpt, 5, "v4")

        v5 = cc(v1, v2, v3, name="v5")
        v6 = cc(v2, v1, v2, v4, name="v6")

        v7 = nc("v7") >> v5 + v6

        g = CompGraph(inpt, v7)
        @test size(g(ones(3))) == (nout(v7),)

        @test minΔnoutfactor(v7) == 2
        @test Δnout(v7, -7)

        @test nout(v1) == 8
        @test nout(v2) == 4
        @test nout(v3) == 6
        @test nout(v4) == 2

        @test size(g(ones(3))) == (nout(v7),)
    end

    @testset "Decorating strategy" begin 
        import NaiveNASlib: DecoratingJuMPΔSizeStrategy

        struct DummyDecorator{S} <: DecoratingJuMPΔSizeStrategy
            s::S
        end
        NaiveNASlib.base(s::DummyDecorator) = s.s

        function graphgen()
            inpt = iv(3)
            v1 = av(inpt, 10, "v1")
            v2 = av(inpt, 5, "v2")
            v3 = av(inpt, 10, "v3")
            v4 = av(inpt, 5, "v4")

            v5 = cc(v1, v2, v3, name="v5")
            v6 = cc(v2, v1, v2, v4, name="v6")

            v7 = nc("v7") >> v5 + v6

            CompGraph(inpt, v7)
        end

        genstrat(::Type{ΔNout{Exact}}, g) = ΔNoutExact(g.outputs[1], -7)
        genstrat(::Type{ΔNout{Relaxed}}, g) = ΔNoutRelaxed(g.outputs[1], -7)
        genstrat(::typeof(ΔNinExact), g) = ΔNinExact(g.outputs[1], (-7, missing))
        
        @testset "$basestrat" for basestrat in (
            ΔNout{Exact},
            ΔNout{Relaxed},
            ΔNinExact,
        )
            g_exp, g_act = graphgen(), graphgen()
            s_exp = genstrat(basestrat, g_exp)
            s_act = genstrat(basestrat, g_act)

            @test Δsize(v -> 1:nout(v), s_exp, g_exp)
            @test Δsize(v -> 1:nout(v), DummyDecorator(s_act), g_act)

            @test nout.(vertices(g_exp)) == nout.(vertices(g_act))
        end
    end

    @testset "Increase SizeStack and SizeInvariant" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(inpt, 4, "v3")
        v4 = av(inpt, 5, "v4")

        v5 = cc(v1, v2, v4, name="v5")
        v6 = cc(v3, v4, v1, name="v6")

        v7 = nc("v7") >> v5 + v6

        g = CompGraph(inpt, v7)
        @test size(g(ones(3))) == (nout(v7),)

        oldg = copy(g)

        @test minΔnoutfactor(v7) == 1
        Δnout(v7, 6)

        @test nout(v1) == 5
        @test nout(v2) == 6
        @test nout(v3) == 6
        @test nout(v4) == 7

        @test size(g(ones(3))) == (nout(v7),)

        # Since we are just increasing we have the same mapping except a few zeros
        @test filter(>(0), g(1:3)) == oldg(1:3)
    end

    @testset "SizeStack increase decrease" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 10, "v2")
        v3 = cc(v1,v2, name="v3")
        v4 = av(v3, 4, "v4")

        g = CompGraph(inpt, v3)
        @test size(g(ones(3))) == (nout(v3),)

        @test Δnout(v -> 1:nout(v), v3 => relaxed(-5), v1 => 2, v2 => relaxed(0))

        @test nin(v3) == [nout(v1), nout(v2)] == [5, 7]
        @test nout(v3) == sum(nin(v3)) == 12

        @test lastins(v3) == [lastouts(v1), lastouts(v2)] == [[1,2,3,-1,-1], [4,5,6,7,8,9,10]]
        @test lastouts(v3) == lastins(v4) == [1,2,3,-1,-1,7,8,9,10,11,12,13] 

        @test size(g(ones(3))) == (nout(v3),)
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
        @test size.(g(ones(3))) == ((nout(v1),), (nout(v9),))

        Δnout(v2, -1)

        @test nout(v6) == 4
        @test nout(v7) == 13
        @test nout(v0) == 9

        @test size.(g(ones(3))) == ((nout(v1),), (nout(v9),))

        Δnout(v2, 5)

        @test nout(v6) == 4
        @test nout(v7) == 18
        @test nout(v0) == 14

        @test size.(g(ones(3))) == ((nout(v1),), (nout(v9),))
    end

    @testset "Increase at vertex removal" begin
        inpt = iv(3)
        v1 = av(inpt, 2, "v1")
        v2 = tv(v1, "v2")
        v3 = av(v2, 4, "v3")
        v4 = av(v3, 3, "v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(3))) == (nout(v4),)

        @test remove!(v3, RemoveStrategy())

        @test lastins(v4)== [1,2,3,4]
        @test lastouts(v1) == lastouts(v2) == [1, 2, -1, -1]

        @test size(g(ones(3))) == (nout(v4),)
    end

    @testset "Decrease at vertex removal" begin
        inpt = iv(3)
        v1 = av(inpt, 2, "v1")
        v2 = tv(v1, "v2")
        v3 = av(v2, 4, "v3")
        v4 = av(v3, 3, "v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(3))) == (nout(v4),)

        @test remove!(v3, RemoveStrategy(DecreaseBigger()))

        # What happened now is that nin(v4) got decreased from 4 to 2. 
        # However, there was absolutely no need at all to select anything from v2 and before as they have not changed.

        @test lastins(v4) == lastouts(v3) == [3, 4]
        @test lastouts(v1) == lastouts(v2) == [1, 2]

        @test size(g(ones(3))) == (nout(v4),)
    end

    @testset "Increase at vertex removal SizeStack" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(v1, 5, "v3")
        v4 = cc(v2, v3, name="v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(3))) == (nout(v4),)

        @test remove!(v3, RemoveStrategy())

        # v4 did not change
        @test lastins(v4) == [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
        @test lastouts(v4) == 1:9

        @test [lastouts(v2), lastouts(v1)] == [[1,2,3,4], [1,2,3,-1,-1]]

        @test size(g(ones(3))) == (nout(v4),)
    end

    @testset "Decrease at vertex removal SizeStack" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = av(inpt, 4, "v2")
        v3 = av(v1, 5, "v3")
        v4 = cc(v2, v3, name="v4")

        g = CompGraph(inpt, v4)
        @test size(g(ones(Float32, 3))) == (nout(v4),)

        @test remove!(v3, RemoveStrategy(DecreaseBigger()))

        @test lastins(v4) == [[1, 2, 3, 4], [2, 4, 5]]
        @test lastouts(v4) == [1, 2, 3, 4, 6, 8, 9] 

        # v2 and v1 did not change
        @test [lastouts(v2), lastouts(v1)] == [[1,2,3,4], [1,2,3]]

        @test size(g(ones(3))) == (nout(v4),)
    end

    @testset "TruncateInIndsToValid" begin
        inpt = iv(3)
        v1 = av(inpt, 3, "v1")
        v2 = tv(v1, "v2")
        v3 = av(v2, 2, "v3")

        g = CompGraph(inpt, v2)
        @test size(g(ones(3))) == (nout(v2),)

        vnew = av(inpt, 7, "vnew")
        create_edge!(vnew, v2;strategy=PostAlign(TruncateInIndsToValid(AlignNinToNout())))
        @test nin(v2) == [nout(v1), nout(vnew)] == [7, 7] 

        @test lastouts(vnew) == 1:7
        @test lastins(v2) == [[1,2,3,-1,-1,-1,-1], [1,2,3,-1,-1,-1,-1]]
        @test lastouts(v2) == [1,2,3,-1,-1,-1,-1]
    end

    @testset "CompConstraint" begin
        struct CompConstraint end
        function NaiveNASlib.compconstraint!(::NaiveNASlib.NeuronIndices, ::AbstractJuMPΔSizeStrategy, ::CompConstraint, data)
            var = data.outselectvars[data.vertex];
            JuMP.@constraint(data.model, var[[1, 3]] .== 0)
        end
        NaiveNASlib.nout(::CompConstraint) = 4
        ccouts = nothing
        NaiveNASlib.Δsize(::CompConstraint, ins::AbstractVector, outs::AbstractVector) = ccouts=outs

        inpt = iv(3)
        v1 = absorbvertex(CompConstraint(), inpt, traitdecoration = tf("v1"))
        v2 = av(v1, 5, "v2")

        Δnout(v1, 3)
        @test ccouts == lastins(v2) == [2, 4, -1, -1, -1, -1, -1]
    end
end
