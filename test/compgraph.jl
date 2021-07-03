import NaiveNASlib:CompGraph, CompVertex, InputVertex, SimpleDiGraph
import LightGraphs:adjacency_matrix,is_cyclic

@testset "Computation graph tests" begin

    @testset "Scalar computation graphs" begin

        # Setup a simple scalar graph which sums two numbers
        ins = InputVertex.(1:3)
        sumvert = CompVertex(+, ins[1], ins[2])
        scalevert = CompVertex(x -> 2x, sumvert)
        graph = CompGraph(inputs(sumvert), [scalevert])
        sumvert2 = CompVertex((x,y) -> x+y+2, ins[1], ins[3])
        graph2out = CompGraph(ins, [scalevert, sumvert2])

        @testset "Structural tests" begin
            @test adjacency_matrix(SimpleDiGraph(graph)) == [0 0 1 0; 0 0 1 0; 0 0 0 1; 0 0 0 0]
            @test adjacency_matrix(SimpleDiGraph(graph2out)) == [0 0 1 0 0 1;
            0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 0 0; 0 0 0 0 0 1; 0 0 0 0 0 0]

            @test nv(graph) == 4
            @test nv(graph2out) == 6

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

        gcopy = copy(graph)

        @test issame(graph, gcopy)
        @test graph(3,4,10) == gcopy(3,4,10)
    end

    @testset "Mutation graph copy" begin
        ins = InputSizeVertex.(InputVertex.(1:3), 1)
        v1 = ins[1] + ins[2]
        v2 = conc(v1, ins[3], dims=1)
        v3 = conc(ins[1], v1, dims=1)
        v4 = v3 - v2
        v5 = ins[1] / v1
        v6 = absorbvertex(identity, v5)
        graph = CompGraph(ins, [v4, v6])

        gcopy = copy(graph)

        @test issame(graph, gcopy)
        @test graph(3,4,10) == gcopy(3,4,10)
    end

    @testset "Graph add trait" begin
        struct MockTrait <: DecoratingTrait
            t::MutationTrait
        end

        inver = inputvertex("in", 3)
        v1 = absorbvertex(+, inver)
        v2 = conc(inver, v1, dims=1)
        graph = CompGraph(inver, v2)

        addtrait(x...;cf) = clone(x...;cf=cf)
        addtrait(t::MutationTrait; cf=cf) = MockTrait(clone(t, cf=clone))

        graphnew = copy(graph, addtrait)

        function testvert(::InputSizeVertex) end
        testvert(v) = @test v.trait isa MockTrait

        foreach(testvert, vertices(graphnew))
    end

    @testset "Graph rename" begin
        v0 = inputvertex("in", 3)
        v1 = absorbvertex(+, v0, traitdecoration = t -> NamedTrait(t, "v1"))
        v2 = conc(v0, v1, dims=1, traitdecoration = t -> NamedTrait(t, "v2"))
        graph = CompGraph(v0, v2)

        rename(x...;cf) = clone(x...;cf=cf)
        rename(s::String; cf) = s * "new"

        graphnew = copy(graph, rename)

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
