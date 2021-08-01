module TinyNNlib 
    using NaiveNASlib
    # A simple linear layer
    mutable struct LinearLayer{T}
        W::Matrix{T}
    end
    # Normally ones uses something like randn here, but this makes output below easier on the eyes
    LinearLayer(nin, nout) = LinearLayer(ones(Int, nout, nin))
    (l::LinearLayer)(x) = l.W * x

    # NaiveNASlib needs to know what LinearLayer considers its output size and input size
    # In this case it is the number of rows and columns of the weight matrix
    # Input size is always a vector since vertices might have multiple inputs
    NaiveNASlib.nin(l::LinearLayer) = [size(l.W, 2)]
    NaiveNASlib.nout(l::LinearLayer) = size(l.W, 1)

    # We also need to tell NaiveNASlib how to change the size of LinearLayer
    # The Δsize function will receive indices to keep from existing weights as well as where to insert new indices
    function NaiveNASlib.Δsize!(l::LinearLayer, newins::AbstractVector, newouts::AbstractVector)
        # newins is a vector of vectors as vertices may have more than one input, but LinearLayer has only one
        # The function NaiveNASlib.parselect can be used to interpret newins and newouts. 
        # We just need to tell it along which dimensions to apply them.
        l.W = NaiveNASlib.parselect(l.W, 1=>newouts, 2=>newins[])
    end

    # Helper function which creates a LinearLayer wrapped in an vertex in a computation graph.
    # This creates a Keras-like API
    linearvertex(in, outsize) = absorbvertex(LinearLayer(nout(in), outsize), in)
    export linearvertex, LinearLayer
end

@testset "README examples" begin

    # Don't forget to update README.md! TODO: Use some real tool to handle this instead...

    @testset "First example" begin
        using NaiveNASlib, Test
        in1 = inputvertex("in1", 1)
        in2 = inputvertex("in2", 1)

        # Create a new vertex which computes the sum of in1 and in2
        # Use >> to attach a name to the vertex
        computation = "add" >> in1 + in2
        @test computation isa NaiveNASlib.AbstractVertex

        # CompGraph helps evaluating the whole graph as a function
        graph = CompGraph([in1, in2], computation);

        # Evaluate the function represented by graph
        @test graph(2,3) == 5
        @test graph(100,200) == 300

        # The vertices function returns the vertices in topological order
        @test vertices(graph) == [in1, in2, computation]
        @test name.(vertices(graph)) == ["in1", "in2", "add"]
    end

    @testset "Second and third examples" begin

        using .TinyNNlib

        # A simple 2 layer model
        invertex = inputvertex("input", 3)
        layer1 = linearvertex(invertex, 4);
        layer2 = linearvertex(layer1, 5);

        # Vertices may be called to execute their computation alone
        batchsize = 2;
        batch = randn(nout(invertex), batchsize);
        y1 = layer1(batch);
        @test size(y1) == (nout(layer1), batchsize) == (4, 2)
        y2 = layer2(y1);
        @test size(y2) == (nout(layer2), batchsize) == (5, 2)

        # This is just because we used nout(in) as the input size when creating the LinearLayer above
        @test [nout(layer1)] == nin(layer2) == [4]

        # Lets change the output size of layer1:
        @test Δnout!(layer1 => -2) # Returns true if successful

        # And now the weight matrices have changed
        @test [nout(layer1)] == nin(layer2) == [2]

        # The graph is still operational :)
        y1 = layer1(batch);
        @test size(y1) == (nout(layer1), batchsize) == (2, 2)
        y2 = layer2(y1);
        @test size(y2) == (nout(layer2), batchsize) == (5 ,2)

        ### Third example ###
        # First a few "normal" layers
        invertex = inputvertex("input", 6);
        start = linearvertex(invertex, 6);
        split = linearvertex(start, nout(invertex) ÷ 3);

        # When multiplying with a scalar, the output size is the same as the input size.
        # This vertex type is said to be size invariant (in lack of better words).
        scalarmult(v, s::Number) = invariantvertex(x -> x .* s, v)

        # Concatenation means the output size is the sum of the input sizes
        joined = conc(scalarmult(split,2), scalarmult(split,3), scalarmult(split,5), dims=1);

        # Elementwise addition is of course also size invariant 
        out = start + joined;

        graph = CompGraph(invertex, out)
        @test graph((ones(6)))  == [78, 78, 114, 114, 186, 186]

        # Ok, lets try to change the size of the vertex "out".
        # Before we do that, lets have a look at the sizes of the vertices in the graph to have something to compare to
        @test [nout(start)] == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [6]
        @test [nout(start), nout(joined)] == nin(out) == [6, 6]

        # In many cases it is useful to hold on to the old graph before mutating
        parentgraph = copy(graph)

        # It is not possible to change the size of out by just 2
        # By default, NaiveNASlib warns when this happens and then tries to make the closest possible change
        # If we don't want the warning, we can tell NaiveNASlib to relax and make the closest possible change right away
        @test Δnout!(out => relaxed(2))

        # We didn't touch the input when mutating...
        @test [nout(invertex)] == nin(start) == [6]
        # Start and joined must have the same size due to elementwise op.
        # All three scalarmult vertices are transparent and propagate the size change to split
        @test [nout(start)] == nin(split) == [3 * nout(split)] == [sum(nin(joined))] == [nout(out)] == [9]
        @test [nout(start), nout(joined)] == nin(out) == [9, 9]

        # parselect used by TinyNNlib will insert zeros when size increases by default. 
        # This helps the graph maintain the same function after mutation
        # In this case we changed the size of the output layer so we don't have the exact same function though
        @test graph((ones(6))) == [78, 78, 0, 114, 114, 0, 186, 186, 0]

        # Copy is still intact
        @test parentgraph((ones(6))) == [78, 78, 114, 114, 186, 186]

        ### More detailed examples ###
        # Supply a utility function for telling the value of each neuron in a vertex
        # NaiveNASlib will prioritize selecting the indices with higher value

        # Prefer high indices:
        graphhigh = copy(graph);
        @test Δnout!(v -> 1:nout(v), graphhigh.outputs[] => -3)
        @test graphhigh((ones(6))) == [42, 0, 60, 0, 96, 0]

        # Perfer low indices
        graphlow = copy(graph);
        @test Δnout!(v -> nout(v):-1:1, graphlow.outputs[] => -3) 
        @test graphlow((ones(6))) == [78, 78, 114, 114, 186, 186]


        # A common approach when doing structured pruning is to prefer neurons with high magnitude.
        # Here is how to set that as the default for LinearLayer.
        # This is something one should probably implement in TinyNNlib instead... 
        import Statistics: mean
        NaiveNASlib.default_outvalue(l::LinearLayer) = mean(abs, l.W, dims=2)

        graphhighmag = copy(graph);
        @test Δnout!(graphhighmag.outputs[] => -3) 
        @test graphhighmag((ones(6))) == [78, 78, 114, 114, 186, 186]

        # In many NAS applications one wants to apply random mutations to the graph
        # When doing so, one might end up in situations like this:
        badgraphdecinc = copy(graph);
        v1, v2 = vertices(badgraphdecinc)[[3, end]]; # Imagine selecting these at random
        @test Δnout!(v1 => relaxed(-2))
        @test Δnout!(v2 => 6)
        # Now we first deleted a bunch of weights, then we added new :(
        @test badgraphdecinc((ones(6))) ==  [42, 0, 0, 60, 0, 0, 96, 0, 0]

        # In such cases, it might be better to supply all wanted changes in one go and let 
        # NaiveNASlib try to come up with a decent compromise.
        goodgraphdecinc = copy(graph);
        v1, v2 = vertices(goodgraphdecinc)[[3, end]];
        @test Δnout!(v1 => relaxed(-2), v2 => 3) # Mix relaxed and exact size changes freely
        @test goodgraphdecinc((ones(6))) == [78, 78, 0, 0, 114, 114, 0, 0, 186, 186, 0, 0] 

        # It is also possible to change the input direction, but it requires specifying a size change for each input
        graphΔnin = copy(graph);
        v1, v2 = vertices(graphΔnin)[end-1:end];
        @test Δnin!(v1 => (3, relaxed(2), missing), v2 => relaxed((1,2))) # Use missing to signal "don't care"
        @test nin(v1) == [6, 6, 6] # Sizes are tied to nout of split so they all have to be equal
        @test nin(v2) == [18, 18] # Sizes are tied due to elementwise addition

        # Another popular pruning strategy is to just remove the x% of params with lowest value
        # This can be done by just not putting any size requirements and assign negative value
        graphprune40 = copy(graph);
        Δsize!(graphprune40) do v
            # Assign no value to SizeTransparent vertices
            NaiveNASlib.trait(v) isa NaiveNASlib.SizeTransparent && return 0
            value = NaiveNASlib.default_outvalue(v)
            return value .- 0.4mean(value)
        end
        @test nout.(vertices(graphprune40)) == [6, 6, 2, 2, 2, 2, 6, 6]
        # Compare to original:
        @test nout.(vertices(graph))        == [6, 9, 3, 3, 3, 3, 9, 9]

        @testset "Weight modification example" begin

            # Return layer just so we can easily look at it
            function vertexandlayer(in, outsize)
                nparam = nout(in) * outsize
                l = LinearLayer(collect(reshape(1:nparam, :, nout(in))))
                return absorbvertex(l, in), l
            end

            # Make a simple model
            invertices = inputvertex.(["in1", "in2"], [3,4])
            v1, l1 = vertexandlayer(invertices[1], 4)
            v2, l2 = vertexandlayer(invertices[2], 3)
            merged = conc(v1, v2, dims=1)
            v3, l3 = vertexandlayer(merged, 2)
            graph = CompGraph(invertices, v3)

            # These weights are of course not useful in a real neural network.
            # They are just to make it easier to spot what has changed after size change below.
            @test l1.W ==
            [ 1 5  9 ; 
              2 6 10 ; 
              3 7 11 ;
              4 8 12 ]

            @test l2.W ==
            [ 1 4 7 10 ;
              2 5 8 11 ; 
              3 6 9 12 ]

            @test l3.W ==
            [ 1  3  5  7   9  11  13 ;
              2  4  6  8  10  12  14 ]

            # Now, lets decrease v1 by 1 and force merged to retain its size 
            # which in turn forces v2 to grow by 1
            # Give high value to neurons 1 and 3 of v2, same for all others...
            @test Δnout!(v2 => -1, merged => 0) do v
                v == v2 ? [10, 1, 10] : ones(nout(v))
            end

            # v1 got a new row of parameters at the end
            @test l1.W ==
            [ 1  5   9 ;
              2  6  10 ;
              3  7  11 ;
              4  8  12 ;
              0  0   0 ]

            # v2 chose to drop its middle row as it was the output neuron with lowest value
            @test l2.W ==
            [ 1 4 7 10 ;
              3 6 9 12 ]

            # v3 dropped the second to last column (which is aligned to the middle row of v2)
            # and got new parameters in column 5 (which is aligned to the last row of v1)
            @test l3.W ==
            [  1  3  5  7  0   9  13 ;
               2  4  6  8  0  10  14 ]
        end


        @testset "Add layers example" begin

            invertex = inputvertex("input", 3)
            layer1 = linearvertex(invertex, 5)
            graph = CompGraph(invertex, layer1)

            # nv(g) is shortcut for length(vertices(g))
            @test nv(graph) == 2
            @test graph(ones(3)) == [3,3,3,3,3]

            # Insert a layer between invertex and layer1
            @test insert!(invertex, vertex -> linearvertex(vertex, nout(vertex))) # True if success

            @test nv(graph) == 3
            @test graph(ones(3)) == [9, 9, 9, 9, 9]
        end

        @testset "Remove layers example" begin
            invertex = inputvertex("input", 3)
            layer1 = linearvertex(invertex, 5)
            layer2 = linearvertex(layer1, 4)
            graph = CompGraph(invertex, layer2)

            @test nv(graph) == 3
            @test graph(ones(3)) == [15, 15, 15, 15]

            # Remove layer1 and change nin of layer2 from 5 to 3
            # Would perhaps have been better to increase nout of invertex, but it is immutable
            @test remove!(layer1) # True if success

            @test nv(graph) == 2
            @test graph(ones(3)) == [3, 3, 3, 3]
        end

        @testset "Add edge example" begin
            invertices = inputvertex.(["input1", "input2"], [3, 2])
            layer1 = linearvertex(invertices[1], 4)
            layer2 = linearvertex(invertices[2], 4)
            add = layer1 + layer2
            out = linearvertex(add, 5)
            graph = CompGraph(invertices, out)

            @test nin(add) == [4, 4]
            # Two inputs this time, remember?
            @test graph(ones(3), ones(2)) == [20, 20, 20, 20, 20]

            # This graph is not interesting enough for there to be a good showcase for adding a new edge.
            # Lets create a new layer which has a different output size just to see how things change
            # The only vertex which support more than one input is add
            layer3 = linearvertex(invertices[2], 6)
            @test create_edge!(layer3, add) # True if success
    
            # By default, NaiveNASlib will try to increase the size in case of a mismatch
            @test nin(add) == [6, 6, 6]
            @test graph(ones(3), ones(2)) == [28, 28, 28, 28, 28] 
        end

        @testset "Remove edge example" begin
            invertex = inputvertex("input", 4)
            layer1 = linearvertex(invertex, 3)
            layer2 = linearvertex(invertex, 5)
            merged = conc(layer1, layer2, layer1, dims=1)
            out = linearvertex(merged, 3)
            graph = CompGraph(invertex, out)

            @test nin(merged) == [3, 5, 3]
            @test graph(ones(4)) == [44, 44, 44]

            @test remove_edge!(layer1, merged) # True if success

            @test nin(merged) == [5, 3]
            @test graph(ones(4)) == [32, 32, 32]
        end

        @testset "Advanced usage" begin
            using NaiveNASlib.Advanced, NaiveNASlib.Extend

            @testset "Strategies" begin
                # A simple graph where one vertex has a constraint for changing the size.
                invertex = inputvertex("in", 3)
                layer1 = linearvertex(invertex, 4)
                # joined can only change in steps of 2
                joined = conc(scalarmult(layer1, 2), scalarmult(layer1, 3), dims=1)          

                # Strategy to try to change it by one and throw an error when not successful
                exact_or_fail = ΔNoutExact(joined => 1; fallback=ThrowΔSizeFailError("Size change failed!!"))

                # Note that we now call Δsize instead of Δnout! as the wanted action is given by the strategy
                @test_throws NaiveNASlib.ΔSizeFailError Δsize!(exact_or_fail)

                # No change was made
                @test nout(joined) == 2*nout(layer1) == 8

                # Try to change by one and fail silently when not successful
                exact_or_noop = ΔNoutExact(joined=>1;fallback=ΔSizeFailNoOp())

                @test !Δsize!(exact_or_noop) 

                # No change was made
                @test nout(joined) == 2*nout(layer1) == 8

                # In many cases it is ok to not get the exact change which was requested
                relaxed_or_fail = ΔNoutRelaxed(joined=>1;fallback=ThrowΔSizeFailError("This should not happen!!"))

                @test Δsize!(relaxed_or_fail)

                # Changed by two as this was the smallest possible change
                @test nout(joined) == 2*nout(layer1) == 10

                # Logging when fallback is applied is also possible
                using Logging
                # Yeah, this is not easy on the eyes, but it gets the job done...
                exact_or_log_then_relax = ΔNoutExact(joined=>1; fallback=LogΔSizeExec("Exact failed, relaxing", Logging.Info, relaxed_or_fail))

                @test_logs (:info, "Exact failed, relaxing") Δsize!(exact_or_log_then_relax)

                @test nout(joined) == 2*nout(layer1) == 12
            end

            @testset "Traits" begin
                noname = linearvertex(inputvertex("in", 2), 2)
                @test name(noname) == "MutationVertex::SizeAbsorb"

                # Naming vertices is so useful for logging and debugging I almost made it mandatory
                named = absorbvertex(LinearLayer(2, 3), inputvertex("in", 2), traitdecoration = t -> NamedTrait(t, "named layer"))
                @test name(named) == "named layer"

                # Speaking of logging...
                layer1 = absorbvertex(LinearLayer(2, 3), inputvertex("in", 2), traitdecoration = t -> SizeChangeLogger(NamedTrait(t, "layer1")))

                # What info is shown can be controlled by supplying an extra argument to SizeChangeLogger
                nameonly = NaiveNASlib.NameInfoStr()
                layer2 = absorbvertex(LinearLayer(nout(layer1), 4), layer1, traitdecoration = t -> SizeChangeLogger(nameonly, NamedTrait(t, "layer2")))

                @test_logs(
                (:info, "Change nout of layer1, inputs=[in], outputs=[layer2], nin=[2], nout=[4], SizeAbsorb() by [1, 2, 3, -1]"),
                (:info, "Change nin of layer2 by [1, 2, 3, -1]"), # Note: less verbose compared to layer1 due to NameInfoStr
                Δnout!(layer1, 1))

                # traitdecoration works exactly the same for conc and invariantvertex as well, no need for an example

                # For more elaborate traits with element wise operations one can use traitconf and >>
                add = traitconf(t -> SizeChangeLogger(NamedTrait(t, "layer1 + layer2"))) >> layer1 + layer2
                @test name(add) == "layer1 + layer2"

                @test_logs(
                (:info, "Change nout of layer1, inputs=[in], outputs=[layer2, layer1 + layer2], nin=[2], nout=[5], SizeAbsorb() by [1, 2, 3, 4, -1]"),
                (:info, "Change nin of layer2 by [1, 2, 3, 4, -1]"),
                (:info, "Change nout of layer2 by [1, 2, 3, 4, -1]"),
                (:info, "Change nin of layer1 + layer2, inputs=[layer1, layer2], outputs=[], nin=[5, 5], nout=[5], SizeInvariant() by [1, 2, 3, 4, -1] and [1, 2, 3, 4, -1]"),
                (:info, "Change nout of layer1 + layer2, inputs=[layer1, layer2], outputs=[], nin=[5, 5], nout=[5], SizeInvariant() by [1, 2, 3, 4, -1]"),
                Δnout!(layer1, 1))

                # When creating own trait wrappers, remember to subtype DecoratingTrait or else there will be pain!

                # Wrong!! Not a subtype of DecoratingTrait
                struct PainfulTrait{T<:MutationTrait} <: MutationTrait
                    base::T
                end
                painlayer = absorbvertex(LinearLayer(2, 3), inputvertex("in", 2), traitdecoration = PainfulTrait)

                # Now one must implement a lot of methods for PainfulTrait...
                @test_throws MethodError Δnout!(painlayer, 1)

                # Right! Is a subtype of DecoratingTrait
                struct SmoothSailingTrait{T<:MutationTrait} <: DecoratingTrait
                    base::T
                end
                # Just implement base and all will be fine
                NaiveNASlib.base(t::SmoothSailingTrait) = t.base

                smoothlayer = absorbvertex(LinearLayer(2, 3), inputvertex("in", 2), traitdecoration = SmoothSailingTrait)

                @test Δnout!(smoothlayer, 1)
                @test nout(smoothlayer) == 4
            end

            @testset "Graph instrumentation and modification" begin
                invertex = inputvertex("in", 2)
                layer1 = linearvertex(invertex, 3)
                layer2 = linearvertex(layer1, 4)

                graph = CompGraph(invertex, layer2)

                @test name.(vertices(graph)) == ["in", "MutationVertex::SizeAbsorb", "MutationVertex::SizeAbsorb"]

                # Ok, lets add names to layer1 and layer2 and change the name of invertex

                # Add a name to layer1 and layer2
                function walk(f, v::MutationVertex)
                    # This is probably not practical to do in a real graph, so make sure you have names when first creating it...
                    name = v == layer1 ? "layer1" : "layer2"
                    addname(x) = x
                    addname(t::SizeAbsorb) = NamedTrait(t, name) # SizeAbsorb has no fields, otherwise we would have had to use walk to wrap it
                    Functors._default_walk(v) do x
                        fmap(addname, x; walk)
                    end
                end

                # Change name of invertex once we get there
                # We could also just have made a string version of addname above since there are no other Strings in the graph, but this is safer
                function walk(f, v::InputVertex)
                    rename(x) = x
                    rename(s::String) = "in changed"
                    Functors._default_walk(v) do x
                        fmap(rename, x; walk)
                    end
                end

                # Everything else just gets functored as normal
                walk(f, x) = Functors._default_walk(f, x) 

                # I must admit that thinking about what this does makes me a bit dizzy...
                namedgraph = fmap(identity, graph; walk)

                @test name.(vertices(namedgraph)) == ["in changed", "layer1", "layer2"]
            end
        end
    end

end
