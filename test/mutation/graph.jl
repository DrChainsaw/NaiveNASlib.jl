@testset "Graph size queries" begin

    # Helper methods
    nt(name) = named(name)
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
            # Also, if we remove v2 then v5 sees v1 through v4 too and we get the impossible size constraint 
            # that nout(v1) == nin(v5) == nout(v4) == nout(v1) + nout(v3b) (where nout(v) > 0 ∀ v).
            v1 = iv(6, "v1")
            v2 = av(v1, 2, "v2")
            v3a = tv(v2, "v3a") 
            v3b = av(inspre(v2), 4, "v3b")
            v4 = cc(v3a, inspost(v3b); name="v4")
            v5 = "v5" >> v4 + v1
            v6 = av(v5, 3, "v6") 

            @testset "Vertex $(name(v))" for v in all_in_graph(v1)
                if v === v3b || v === v2
                    @test isinsizecycle(v) == true
                else
                    @test isinsizecycle(v) == false
                end
            end
        end

        @testset "Simple cycle with inputvertices" begin
            v1 = iv(6, "v1")
            v2 = iv(2, "v2")
            v3a = tv(v2, "v3a") 
            v3b = av(v2, 4, "v3b")
            v4 = cc(v3a, v3b; name="v4")
            v5 = "v5" >> v4 + v1
            v6 = av(v5, 3, "v6") 
            
            @testset "Vertex $(name(v))" for v in all_in_graph(v1)
                if v === v3b # v2 is not in sizecycle as it does not have ancestors
                    @test isinsizecycle(v) == true
                else
                    @test isinsizecycle(v) == false
                end
            end     
        end

        @testset "Double absorbing in path -> no cycle" begin
            # Same as above, but now there are two size absorbing vertices in the non-transparent path
            # v3b1 does not see v5 as v3b2 blocks the line of sight and v3b2 does not see v2 as v3b1 block the line of sight
            v1 = iv(6, "v1")
            v2 = av(v1, 2, "v2")
            v3a = tv(v2, "v3a") 
            v3b1 = av(v2, 2, "v3b1")
            v3b2 = av(v3b1, 4, "v3b2")
            v4 = cc(v3a, v3b2; name="v4")
            v5 = "v5" >> v4 + v1
            v6 = av(v5, 3, "v6") 

            @testset "Vertex $(name(v))" for v in all_in_graph(v1)
                if v === v2
                    @test isinsizecycle(v) == true
                else
                    @test isinsizecycle(v) == false
                end
            end
        end

        @testset "Independent concats" begin
            # Counterexample to Simple cycle. Not a size cycle as the two paths go through different concatenations
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

            @test remove!(v3a) == true
            @test isinsizecycle(v3b) == true
        end

        @testset "Concat and elemwise connected in cycle" begin
            v1 = iv(2, "v1")
            v2 = av(v1, 6, "v2")
            v3a = av(v2, 2, "v3a")
            v3b1 = av(v2, 3, "v3b1")
            v3b2 = av(v3b1, 4, "v3b2")
            v4 = cc(v3a, v3b2; name="v4")
            v5 = "v5" >> v4 + v2
            v6 = av(v5, 3, "v6") 

            @testset "Vertex $(name(v))" for v in all_in_graph(v1)
                if v === v3a
                    @test isinsizecycle(v) == true
                else
                    @test isinsizecycle(v) == false
                end
            end
        end

        @testset "Size cycle when adding edge" begin
            v1 = av(iv(3, "in"), 3, "v1")
            v2 = av(v1, 5, "v2")
            v3 = tv(v2, "v3")
            ve = tv(v1, "ve")
            v4 = cc(v3; name="v4")
            v5 = "v5" >> v4 + v2

            @testset "Vertex $(name(v))" for v in all_in_graph(v1)
                @test isinsizecycle(v) == false
            end 

            # Note: this is basically the default when adding edges to SizeStack
            # We first add it, and then we try to align the sizes
            # In this case alignment becomes impossible due to the created size cycle
            create_edge!(ve, v4; strategy=NoSizeChange())

            @testset "Vertex $(name(v)) after edge" for v in all_in_graph(v1)
                if v === v2 || v === v3
                    @test isinsizecycle(v) == true
                else
                    @test isinsizecycle(v) == false
                end
            end
        end
    end
end
