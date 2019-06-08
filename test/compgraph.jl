import NaiveNASlib:CompGraph, CompVertex, InputVertex, SimpleDiGraph
import LightGraphs:adjacency_matrix,is_cyclic
using Test

@testset "Computation graph tests" begin

    @testset "Scalar computation graphs" begin

        # Setup a simple scalar graph which sums two numbers
        ins = InputVertex.(1:3)
        sumvert = CompVertex(+, ins[1], ins[2])
        scalevert = CompVertex(x -> 2x, sumvert)
        graph = CompGraph(sumvert.inputs, [scalevert])
        sumvert2 = CompVertex((x,y) -> x+y+2, ins[1], ins[3])
        graph2out = CompGraph(ins, [scalevert, sumvert2])

        @testset "Structural tests" begin
            @test adjacency_matrix(SimpleDiGraph(graph)) == [0 0 1 0; 0 0 1 0; 0 0 0 1; 0 0 0 0]
            @test adjacency_matrix(SimpleDiGraph(graph2out)) == [0 0 1 0 0 0;
            0 0 1 0 0 1; 0 0 0 1 0 0; 0 0 0 0 0 0; 0 0 0 0 0 1; 0 0 0 0 0 0]
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
        v1 = AbsorbVertex(CompVertex(+, ins[1], ins[2]), IoSize(1,1))
        v2 = StackingVertex(CompVertex(vcat, v1, ins[3]))
        v3 = StackingVertex(CompVertex(vcat, ins[1], v1))
        v4 = AbsorbVertex(CompVertex(-, v3, v2), IoSize(1,1))
        v5 = InvariantVertex(CompVertex(/, ins[1], v1))
        graph = CompGraph(ins, [v4, v5])
        #TODO outputs as [v5, v4] causes graphs to not be identical
        # This is due to v5 being a mostly independent branch
        # which is the completed before the v4 branch
        # I think this does not matter in practice (as the branches
        #  are independent), but in this test case we are testing
        # for identity

        gcopy = copy(graph)

        @test issame(graph, gcopy)
        @test graph(3,4,10) == gcopy(3,4,10)

        newop(v::MutationVertex) = newop(trait(v), v)
        newop(::MutationTrait, v::MutationVertex) = clone(op(v))
        newop(::SizeAbsorb, v::MutationVertex) = IoIndices(nin(v), nout(v))
        graph_inds = copy(graph, newop)

        @test !issame(graph_inds, graph)
        @test graph(3,4,10) == graph_inds(3,4,10)

        # Nothing should have changed with original
        function testop(v) end
        testop(v::MutationVertex) = testop(trait(v), v)
        function testop(::MutationTrait, v) end
        testop(::SizeAbsorb, v) = @test typeof(op(v)) == IoSize
        foreach(testop, mapreduce(flatten, vcat, graph.outputs))

        # But new graph shall use IoIndices
        testop(::SizeAbsorb, v) = @test typeof(op(v)) == IoIndices
        foreach(testop, mapreduce(flatten, vcat, graph_inds.outputs))

    end
end
