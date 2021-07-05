@testset "Graphs" begin
    using LightGraphs, MetaGraphs

    # Helper methods
    nt(name) = t -> NamedTrait(t, name)
    tf(name) = t -> nt(name)(t)
    iv(size, name="in") = inputvertex(name, size)
    tv(in, name) = invariantvertex(identity, in, traitdecoration=tf(name))
    av(in, outsize, name) = absorbvertex(IndMem(MatMul(nout(in), outsize)), in, traitdecoration=tf(name))
    cc(ins...; name) = conc(ins...; dims=2, traitdecoration=tf(name), outwrap=x -> IndMem(x, collect(nout.(ins)), sum(nout, ins)))

    nc(name) = traitconf(nt(name))

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
            Δnin(v3, 2, 1)

            clist = changed(all_in_graph(v0))
            @test Set(clist) == Set(NaiveNASlib.vertexproplist(g, :vertex))

            for e in edges(g)
                testedge(get_prop(g, e, :direction), g[e.src,:vertex], g[e.dst, :vertex], get_prop(g, e, :size))
            end
        end
    end
end
