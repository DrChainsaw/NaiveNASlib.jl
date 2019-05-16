import NaiveNASlib:CompGraph, CompVertex, InputVertex, SimpleDiGraph
import LightGraphs:adjacency_matrix,is_cyclic
using Test

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
