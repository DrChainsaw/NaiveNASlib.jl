@testset "Computation graph tests" begin

    using Functors: fmap

    @testset "Scalar computation graphs" begin

        # Setup a simple scalar graph which sums two numbers
        ins = InputVertex.(1:3)
        sumvert = CompVertex(+, ins[1], ins[2])
        scalevert = CompVertex(x -> 2x, sumvert)
        graph = CompGraph(inputs(sumvert), [scalevert])
        sumvert2 = CompVertex((x,y) -> x+y+2, ins[1], ins[3])
        graph2out = CompGraph(ins, [scalevert, sumvert2])

        @testset "Structural tests" begin
            @test nvertices(graph) == 4
            @test nvertices(graph2out) == 6

            @test vertices(graph) == [ins[1], ins[2], sumvert, scalevert]
            @test vertices(graph2out) == [ins[1], ins[2], sumvert, scalevert, ins[3], sumvert2]
        end

        @testset "Computation tests" begin
            @test graph(2,3) == 10
            @test graph([2], [3]) == [10]
            @test graph(0.5, 1.3) ≈ 3.6
            @test graph2out(4,5,8) == (18, 14)
        end
    end

    @testset "Array computation graphs" begin

        # Setup a graph which scales one of the inputs by 3 and then merges is with the other
        ins = InputVertex.(1:2)
        scalevert = CompVertex(x -> 3 .* x, ins[1])
        mergegraph1 = CompGraph(ins, [CompVertex(hcat, ins[2], scalevert)])
        mergegraph2 = CompGraph(ins, [CompVertex(hcat, scalevert, ins[2])])

        @testset "Computation tests" begin
            @test CompGraph([ins[1]], [scalevert])(ones(Int64, 1, 2)) == [3 3]

            @test mergegraph1(ones(Int64, 3,2), ones(Int64, 3,3)) == [1 1 1 3 3; 1 1 1 3 3; 1 1 1 3 3]
            @test mergegraph2(ones(Int64, 3,2), ones(Int64, 3,3)) == [3 3 1 1 1; 3 3 1 1 1; 3 3 1 1 1]
        end

    end

    @testset "Simple graph copy" begin
        ins = InputVertex.(1:3)
        v1 = CompVertex(+, ins[1], ins[2])
        v2 = CompVertex(vcat, v1, ins[3])
        v3 = CompVertex(vcat, ins[1], v1)
        v4 = CompVertex(-, v3, v2)
        v5 = CompVertex(/, ins[1], v1)
        graph = CompGraph(ins, [v5, v4])
        
        @testset "Copy graph with $label" for (label, cfun) in (
            (deepcopy, deepcopy),
            ("fmap", g -> fmap(identity, g))
        )
            gcopy = cfun(graph)

            @test issame(graph, gcopy)
            @test graph(3,4,10) == gcopy(3,4,10)
        end
    end

    @testset "Indexing" begin
        in1 = inputvertex("in1", 1)
        in2 = inputvertex("in2", 1)
        v1 = absorbvertex(MatMul(nout(in1), nout(in1)), in1; traitdecoration=named("v1"))
        v2 = "v2" >> in2 + v1
        v3 = absorbvertex(MatMul(nout(v2), 2), v2; traitdecoration=named("v3"))
        v4 = conc(v1, v3; dims=2)

        graph = CompGraph([in1, in2], v4)

        @test graph[1:nvertices(graph)] == graph[begin:end] == vertices(graph)
        @test graph[begin] == graph[1] == in1
        @test graph[nvertices(graph)] == graph[end] == v4

        @test findvertices(graph, "v1") == [v1]
        @test findvertices(graph, r"^in") == [in1, in2]

        @test graph.v2 == v2

    end

    @testset "Mutation graph copy" begin
        ins = inputvertex.(1:3, 1)
        v1 = ins[1] + ins[2]
        v2 = conc(v1, ins[3], dims=1)
        v3 = conc(ins[1], v1, dims=1)
        v4 = v3 - v2
        v5 = ins[1] / v1
        v6 = absorbvertex(identity, v5)
        graph = CompGraph(ins, [v4, v6])
        vs = vertices(graph)

        @testset "Copy graph with $label" for (label, cfun) in (
            (deepcopy, deepcopy),
            ("fmap", g -> fmap(identity, g))
        )
            gcopy = cfun(graph)

            @test issame(graph, gcopy)
            @test graph(3,4,10) == gcopy(3,4,10)

            vscopy = vertices(gcopy)
            @test length(vscopy) == length(vs)
            @test any(v -> v ∈ vs, vscopy) == false

            allgraph = all_in_graph(inputs(gcopy)[1])
            @test length(allgraph) == length(vs)
            @test any(v -> v ∈ vs, allgraph) == false

            @testset "Consistency for vertex $i" for (i, v) in enumerate(vscopy)
                @testset "Is in outputs of input $j" for (j, vi) in enumerate(inputs(v))
                    @test v in outputs(vi)
                end
                @testset "Is in inputs of output $j" for (j, vo) in enumerate(outputs(v))                  
                    @test v in inputs(vo)
                end
            end
        end
    end

    @testset "Graph add/remove trait" begin
        struct MockTrait <: DecoratingTrait
            t::MutationTrait
        end

        @functor MockTrait

        inver = inputvertex("in", 3)
        v1 = absorbvertex(+, inver)
        v2 = conc(inver, v1, dims=1)
        graph = CompGraph(inver, v2)

        # slightly annoying that Functors treats leaves and non-leaves differently
        # MutationTraits happen to be leaves as they have no children
        addtrait(x) = x
        addtrait(t::MutationTrait) = MockTrait(t)
        graphnewtrait = fmap(addtrait, graph)

        @testset "Check trait for vertex $i" for (i, v) in enumerate(filter(v -> v ∉ inputs(graphnewtrait), vertices(graphnewtrait)))
            @test trait(v) isa MockTrait
        end

        rmtrait(f, x) = Functors._default_walk(f, x)
        rmtrait(f, t::MockTrait) = t.t
        grapholdtrait = fmap(identity, graphnewtrait; walk=rmtrait)
       
        expected = (SizeAbsorb, SizeStack)
        @testset "Check trait for vertex $i" for (i, v) in enumerate(filter(v -> v ∉ inputs(grapholdtrait), vertices(grapholdtrait)))
            @test trait(v) isa expected[i]
        end
    end

    @testset "Graph rename" begin
        v0 = inputvertex("in", 3)
        v1 = absorbvertex(+, v0, traitdecoration = t -> NamedTrait(t, "v1"))
        v2 = conc(v0, v1, dims=1, traitdecoration = t -> NamedTrait(t, "v2"))
        graph = CompGraph(v0, v2)

        rename(x) = x
        rename(s::String) = s * "new"

        graphnew = fmap(rename, graph)

        @test name.(vertices(graphnew)) == ["innew", "v1new", "v2new"]
    end

    @testset "Topological sort" begin
        in1,in2 = InputVertex.(("in1", "in2"))
        v1 = CompVertex(+, in1, in2)
        v2 = CompVertex(-, in2, in1)
        v3 = CompVertex(vcat, v1,v2,v1)
        g = CompGraph([in1, in2], v3)

        @test vertices(g) == [in1, in2, v1, v2, v3]
    end
end
