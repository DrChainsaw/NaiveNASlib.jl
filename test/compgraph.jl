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
            @test adjacency_matrix(SimpleDiGraph(graph2out)) == [0 0 1 0 0 0;
            0 0 1 0 0 1; 0 0 0 1 0 0; 0 0 0 0 0 0; 0 0 0 0 0 1; 0 0 0 0 0 0]

            @test nv(graph) == 4
            @test nv(graph2out) == 6
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

    @testset "README examples" begin

        @testset "First example" begin
            in1, in2 = InputVertex.(("in1", "in2"));

            computation = CompVertex(+, in1, in2);

            graph = CompGraph([in1, in2], computation);

            using Test

            @test graph(2,3) == 5
        end

        @testset "Second and third examples" begin

            # First we need something to mutate. Batteries excluded, remember?
            mutable struct SimpleLayer
                W
                SimpleLayer(W) = new(W)
                SimpleLayer(nin, nout) = new(ones(Int, nin,nout))
            end
            (l::SimpleLayer)(x) = x * l.W

            # Helper function which creates a mutable layer.
            layer(in, outsize) = absorbvertex(SimpleLayer(nout(in), outsize), outsize, in, mutation=IoSize)

            invertex = inputvertex("input", 3)
            layer1 = layer(invertex, 4);
            layer2 = layer(layer1, 5);

            @test [nout(layer1)] == nin(layer2) == [4]

            #Lets change the output size of layer1:
            Δnout(layer1, -2);

            @test [nout(layer1)] == nin(layer2) == [2]

            ### Third example ###
            scalarmult(v, f::Integer) = vertex(x -> x .* f, nout(v), SizeInvariant(), v)

            invertex = inputvertex("input", 6);
            start = layer(invertex, 6);
            split = layer(start, div(nout(invertex) , 3));
            joined = conc(scalarmult(split, 2), scalarmult(split,3), scalarmult(split,5), dims=2);
            out = start + joined;

            @test [nout(invertex)] == nin(start) == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [6]
            @test [nout(start), nout(joined)] == nin(out) == [6, 6]

            graph = CompGraph(invertex, out)
            @test graph((ones(Int, 1,6))) == [78  78  114  114  186  186]

            # Ok, lets try to reduce the size of the vertex "out".
            # First we need to realize that we can only change it by integer multiples of 3
            # This is because it is connected to "split" through three paths which require nin==nout

            # We need this information from the layer. Some layers have other requirements
            NaiveNASlib.minΔnoutfactor(::SimpleLayer) = 1
            NaiveNASlib.minΔninfactor(::SimpleLayer) = 1

            @test minΔnoutfactor(out) == minΔninfactor(out) == 3

            # Next, we need to define how to mutate our SimpleLayer
            NaiveNASlib.mutate_inputs(l::SimpleLayer, newInSize) = l.W = ones(Int, newInSize, size(l.W,2))
            NaiveNASlib.mutate_outputs(l::SimpleLayer, newOutSize) = l.W = ones(Int, size(l.W,1), newOutSize)

            #In some cases it is useful to hold on to the old graph before mutating
            # To do so, we need to define the clone operation for our SimpleLayer
            NaiveNASlib.clone(l::SimpleLayer) = SimpleLayer(l.W)
            parentgraph = copy(graph)

            Δnin(out, 3)

            # We didn't touch the input when mutating...
            @test [nout(invertex)] == nin(start) == [6]
            # Start and joined must have the same size due to elementwise op.
            # All three scalarmult vertices are transparent and propagate the size change to split
            @test [nout(start)] == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [9]
            @test [nout(start), nout(joined)] == nin(out) == [9, 9]

            # However, this only updated the mutation metadata, not the actual layer.
            # There are some slightly annoying and perhaps overthought reasons to this
            # I will document them once things crystalize a bit more
            @test graph((ones(Int, 1,6))) == [78  78  114  114  186  186]

            # To mutate the graph, we need to apply the mutation:
            apply_mutation(graph);

            @test graph((ones(Int, 1,6))) == [114  114  114  168  168  168  276  276  276]

            # Copy is still intact
            @test parentgraph((ones(Int, 1,6))) == [78  78  114  114  186  186]

            @testset "Add layers example" begin

                invertex = inputvertex("input", 3)
                layer1 = layer(invertex, 5)
                graph = CompGraph(invertex, layer1)

                @test nv(graph) == 2
                @test graph(ones(Int, 1, 3)) == [3 3 3 3 3]

                # Insert a layer between invertex and layer1
                insert!(invertex, vertex -> layer(vertex, nout(vertex)))

                @test nv(graph) == 3
                @test graph(ones(Int, 1, 3)) == [9 9 9 9 9]
            end

            @testset "Remove layers example" begin
                invertex = inputvertex("input", 3)
                layer1 = layer(invertex, 5)
                layer2 = layer(layer1, 4)
                graph = CompGraph(invertex, layer2)

                @test nv(graph) == 3
                @test graph(ones(Int, 1, 3)) == [15 15 15 15]

                # Remove layer1 and change nin of layer2 from 5 to 3
                # Would perhaps have been better to increase nout of invertex, but it is immutable
                remove!(layer1)
                apply_mutation(graph)

                @test nv(graph) == 2
                @test graph(ones(Int, 1, 3)) == [3 3 3 3]
            end

            @testset "Add edge example" begin
                invertices = inputvertex.(["input1", "input2"], [3, 2])
                layer1 = layer(invertices[1], 4)
                layer2 = layer(invertices[2], 4)
                add = layer1 + layer2
                out = layer(add, 5)
                graph = CompGraph(invertices, out)

                @test nin(add) == [4, 4]
                # Two inputs this time, remember?
                @test graph(ones(Int, 1, 3), ones(Int, 1, 2)) == [20 20 20 20 20]

                # This graph is not interesting enough for there to be a good showcase for adding a new edge.
                # Lets create a new layer which has a different output size just to see how things change
                # The only vertex which support more than one input is add
                layer3 = layer(invertices[2], 6)
                create_edge!(layer3, add)
                apply_mutation(graph)

                # By default, NaiveNASlib will try to increase the size in case of a mismatch
                @test nin(add) == [6, 6, 6]
                @test graph(ones(Int, 1, 3), ones(Int, 1, 2)) == [42 42 42 42 42]
            end

            @testset "Remove edge example" begin
                invertex = inputvertex("input", 4)
                layer1 = layer(invertex, 3)
                layer2 = layer(invertex, 5)
                merged = conc(layer1, layer2, layer1, dims=2)
                out = layer(merged, 3)
                graph = CompGraph(invertex, out)

                @test nin(merged) == [3, 5, 3]
                @test graph(ones(Int, 1, 4)) == [44 44 44]

                remove_edge!(layer1, merged)
                apply_mutation(graph)

                @test nin(merged) == [5, 6]
                @test graph(ones(Int, 1, 4)) == [44 44 44]
            end
        end
    end
end
