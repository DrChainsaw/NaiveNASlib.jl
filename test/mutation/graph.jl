@testset "Graphs" begin
    using LightGraphs, MetaGraphs
    using NaiveNASlib: SizeDiGraph, fullgraph

    # Helper methods
    nt(name) = t -> NamedTrait(t, name)
    tf(name) = t -> nt(name)(t)
    iv(size, name="in") = inputvertex(name, size)
    tv(in, name) = invariantvertex(identity, in, traitdecoration=tf(name))
    av(in, outsize, name) = absorbvertex(IndMem(MatMul(nout(in), outsize)), in, traitdecoration=tf(name))
    cc(ins...; name) = conc(ins...; dims=2, traitdecoration=tf(name), outwrap=x -> IndMem(x, collect(nout.(ins)), sum(nout, ins)))

    nc(name) = traitconf(nt(name))

    @testset "all_in_Δsize_graph" begin
        using NaiveNASlib: TightΔSizeGraph, LooseΔSizeGraph, all_in_Δsize_graph, Input, Output

        @testset "Concat and elemwise not connected" begin
            v1 = iv(3, "v1")
            v2a = av(v1, 2, "v2a")
            v2b = av(v1, 4, "v2b")
            v3 = cc(v2a, v2b; name="v3")
            v4a = av(v3, 2, "v4a")
            v4b = av(v3, 2, "v4b")
            v5 = "v5" >> v4a + v4b
            v6 = av(v5, 3, "v6") 

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v2a, Output())) == ["v2a", "v3", "v2b", "v4a", "v4b"]
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v2a, Output())) == ["v2a", "v3", "v4a", "v4b"]

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v2a, Input())) == ["v2a", "v1"]
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v2a, Input())) == ["v2a", "v1"]

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v3, Output())) == ["v3", "v2a", "v2b", "v4a", "v4b"]
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v3, Output())) == ["v3", "v2a", "v4a", "v4b", "v2b"] 

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v3, Input())) == ["v3", "v2a", "v2b", "v4a", "v4b"]
            # When we start from v3, at least one of v2a and v2b must change, and we decide to add both in the set, perhaps we should just pick one?
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v3, Input())) == ["v3", "v2a", "v2b", "v4a", "v4b"]

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v4a, Output())) == ["v4a", "v5", "v4b", "v6"] 
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v4a, Output())) == ["v4a", "v5", "v4b", "v6"] 

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v4a, Input())) == ["v4a", "v3", "v2a", "v2b", "v4b"] 
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v4a, Input())) == ["v4a", "v3", "v2a", "v4b", "v2b"] 

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v5, Output())) == ["v5", "v4a", "v4b", "v6"]
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v5, Output())) == ["v5", "v4a", "v4b", "v6"]

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v5, Input())) == ["v5", "v4a", "v4b", "v6"]
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v5, Input())) == ["v5", "v4a", "v4b", "v6"]
        end

        @testset "Concat and elemwise connected" begin
            v1 = iv(6, "v1")
            v2 = av(v1, 2, "v2")
            v3a = av(v2, 2, "v3a")
            v3b = av(v2, 4, "v3b")
            v4 = cc(v3a, v3b; name="v4")
            v5 = "v5" >> v4 + v1
            v6 = av(v5, 3, "v6") 

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v3a, Output())) == ["v3a", "v4", "v3b", "v5", "v1", "v6"]
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v3a, Output())) == ["v3a", "v4", "v5", "v1", "v6"]

            @test name.(all_in_Δsize_graph(LooseΔSizeGraph(), v4, Output())) == ["v4", "v3a", "v3b", "v5", "v1", "v6"]
            # When we start from v4, at least one of v3a and v3b must change, and we decide to add both in the set, perhaps we should just pick one?
            @test name.(all_in_Δsize_graph(TightΔSizeGraph(), v4, Output())) == ["v4", "v3a", "v5", "v1", "v6", "v3b"] 
        end
    end

    @testset "SizeCycleDetector" begin
        using NaiveNASlib: isinsizecycle

        @testset "Simple cycle$label" for (label, inspre, inspost) in (
            ("", identity, identity),
            (" transparent before", v -> tv(v, "tvpre"), identity),
            (" transparent after", identity, v -> tv(v, "tvpost")),
            (" transparent before and after", v -> tv(v, "tvpre"),v -> tv(v, "tvpost")),
            (" concat before", v -> cc(v; name="ccpre"), identity),
            (" concat after", identity, v -> cc(v;name="ccpost")),
            (" concat before and after", v -> cc(v; name="ccpre"), v -> cc(v;name="ccpost")),
        )
            # Canonical size cycle: if we remove v3b we get the constraint that nout(v2) == nin(v5) == nout(v4) == 2 * nout(v2)
            v1 = iv(6, "v1")
            v2 = av(v1, 2, "v2")
            v3a = tv(v2, "v3a") 
            v3b = av(inspre(v2), 4, "v3b")
            v4 = cc(v3a, inspost(v3b); name="v4")
            v5 = "v5" >> v4 + v1
            v6 = av(v5, 3, "v6") 

            @testset "Vertex $(name(v))" for v in all_in_graph(v1)
                if v == v3b
                    @test isinsizecycle(v) == true
                else
                    @test isinsizecycle(v) == false
                end
            end
        end

        @testset "Independent concats" begin
            # Counterexample to the above. Not a size cycle as the two paths go through different concatenations
            v1 = iv(2, "v1")
            v2 = av(v1, 2, "v2")
            v3a = tv(v2, "v3a")
            v3b = av(v2, 2, "v3b")
            v4a = cc(v3a; name="v4a")
            v4b = cc(v3b; name="v4b")
            v5 = "v5" >> v4a + v4b
            v6 = av(v5, 3, "v6") 

            @testset "Vertex $(name(v))" for v in all_in_graph(v1)
                @test isinsizecycle(v) == false
            end
        end


        @testset "Concat and elemwise connected no cycle" begin
            v1 = iv(6, "v1")
            v2 = av(v1, 2, "v2")
            v3a = av(v2, 2, "v3a")
            v3b = av(v2, 4, "v3b")
            v4 = cc(v3a, v3b; name="v4")
            v5 = "v5" >> v4 + v1
            v6 = av(v5, 3, "v6") 

            @testset "Vertex $(name(v))" for v in all_in_graph(v1)
                @test isinsizecycle(v) == false
            end
        end
    end

    @testset "SizeDiGraph" begin

        @testset "SizeAbsorb" begin
            v0 = iv(3)
            v1 = av(v0, 4, "v1")
            v2 = av(v1, 5, "v2")

            g = SizeDiGraph(v1)

            @test nv(g) == 2
            @test ne(g) == 1

            g = fullgraph(v0)

            @test nv(g) == 3
            @test ne(g) == 2

            @test g[1,:vertex] == v0
            @test g[2,:vertex] == v1

            for e in edges(g)
                @test nout(g[e.src,:vertex]) == get_prop(g, e, :size)
            end
        end

        @testset "SizeStack" begin
            v0 = iv(3, "in1")
            v1 = iv(4, "in2")
            v2 = cc(v0, v1, name="v2")

            g = fullgraph(v0)

            @test nv(g) == 3
            @test ne(g) == 2

            @test g[1,:vertex] == v0
            @test g[2,:vertex] == v2

            for e in edges(g)
                @test nout(g[e.src,:vertex]) == get_prop(g, e, :size)
            end
        end

        @testset "SizeInvariant" begin
            v0 = iv(3)
            v1 = tv(v0, "v1")
            v2 = nc("v2") >> v0 + v1

            g = fullgraph(v0)

            @test nv(g) == 3
            @test ne(g) == 3

            @test g[1,:vertex] == v0
            @test g[2,:vertex] == v1
            @test g[3,:vertex] == v2

            for e in edges(g)
                @test nout(g[e.src,:vertex]) == get_prop(g, e, :size)
            end
        end
    end

    @testset "ΔSizeGraph" begin
        using NaiveNASlib: Input, Output, ΔninSizeGraph, ΔnoutSizeGraph
        changed(vs) = filter(vi -> trait(vi) !== Immutable() && (ismissing(lastins(vi)) || any(<(1), lastouts(vi)) || any(nins -> any(<(1), nins), lastins(vi))), vs)

        function testedge(::Input, src, dst, size)
            @test count(>(0), lastouts(src)) == size
            instocheck = lastins(dst)
            if instocheck isa Vector{Int}
                @test count(>(0), instocheck) == size
            else
                @test map(ins -> count(>(0), ins), instocheck[inputs(dst) .== src]) == [size]
            end
        end
        function testedge(::Output, src, dst, size)
            @test count(>(0), lastouts(dst)) == size
            instocheck = lastins(src)
            if instocheck isa Vector{Int}
                @test count(>(0), instocheck) == size
            else
                @test map(ins -> count(>(0), ins), instocheck[inputs(src) .== dst]) == [size]
            end
        end

        @testset "ΔnoutSizeGraph" begin
            v0 = iv(3)
            v1 = av(v0, 4, "v1")
            v2 = av(v0, 5, "v2")
            v3 = cc(v1, v2, name="v3")
            v4 = av(v1, 3, "v4")
            v5 = av(v3, 3, "v5")
            v6 = VertexConf(nt("v6"), f -> IndMem(f, [nout(v4)], nout(v4))) >> v4 + v5
            v7 = cc(v6, v2, name="v7")

            g = ΔnoutSizeGraph(v5)
            Δnout!(v5, 2)

            clist = changed(all_in_graph(v0))
            glist = NaiveNASlib.vertexproplist(g, :vertex)
            for v in clist
                @test v in glist
            end

            for e in edges(g)
                testedge(get_prop(g, e, :direction), g[e.src,:vertex], g[e.dst, :vertex], get_prop(g, e, :size))
            end
        end

        @testset "ΔninSizeGraph" begin
            v0 = iv(3)
            v1 = av(v0, 4, "v1")
            v2 = av(v0, 5, "v2")
            v3 = cc(v1, v2, name="v3")
            v4 = av(v0, 9, "v4")
            v5 = VertexConf(nt("v5"), f -> IndMem(f, [nout(v4)], nout(v4)))  >> v4 + v3
            v6 = av(v5, 3, "v6")

            g = ΔninSizeGraph(v3)
            Δnin!(v3, 2, 1)

            clist = changed(all_in_graph(v0))
            @test Set(clist) == Set(NaiveNASlib.vertexproplist(g, :vertex))

            for e in edges(g)
                testedge(get_prop(g, e, :direction), g[e.src,:vertex], g[e.dst, :vertex], get_prop(g, e, :size))
            end
        end
    end
end
