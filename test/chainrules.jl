@testset "ChainRules" begin

    # We don't implement any primitive rrules in here, so we just test that the plumbing works with some popular AD
    # Therefore ChainRulesTestUtils does not seem useful
    @testset "Zygote" begin
        import Zygote
        gradient = Zygote.gradient

        @testset "Simple ops" begin
            
            vi1,vi2 = inputvertex.("in", (1,1))
            v1 = "add" >> vi1 + vi2
            v2 = "mul" >> vi1 * v1
            v3 = "div" >> v1 / v2

            @test gradient(v1, 2.0, 3.0) == gradient(+, 2.0, 3.0)
            @test gradient(v -> v(2.0, 3.0), v1) == (nothing,)
            
            graph = CompGraph([vi1, vi2], v3)
            function fgraph(vi1,vi2) 
                v1 = vi1 + vi2
                v2 = vi1 * v1
                v3 = v1 / v2
            end

            @test gradient(graph, 2.0, 3.0) == gradient(fgraph, 2.0, 3.0)
            # No parameters
            @test gradient(g -> g(2.0, 3.0), graph) == (nothing,)
        end

        @testset "With parameters" begin
            # We have a MatMul which could be used, but it is mutable so https://github.com/FluxML/Zygote.jl/issues/1111
            # prevents testing of gradients
            struct ImMatMul{M<:AbstractMatrix}
                W::M
            end
            ImMatMul(nin, nout) = ImMatMul(reshape(collect(1:nin*nout), nout,nin))
            NaiveNASlib.nout(mm::ImMatMul) = size(mm.W, 1)
            @functor ImMatMul
            
            (mm::ImMatMul)(x) = mm.W * x

           testgrads(g::CompGraph, res, exp; seen=Base.IdSet()) = foreach(enumerate(outputs(g))) do (i, vo)
                testgrads(vo, seen, res.outputs[i] ,exp)
           end

            function testgrads(v::AbstractVertex, seen, res, exp) 
                v in seen && return
                push!(seen, v)
                _testgrads(v, seen, res, exp, Symbol(name(v)))
            end

            function _testgrads(::InputSizeVertex, seen, res, exp, name) end

            function _testgrads(v::AbstractVertex, seen, res::RT, exp, name) where RT 
                # Or else the gradient is nothing. Previous Dict mutation approach resulted in the full fieldname 
                # tree with all values set to nothing (e.g. (base=(computation=nothing, inputs=[nothing, nothing])).
                # Not sure if that was better or worse than what it is now.
                if computation(v) isa ImMatMul
                    @testset "Check gradient structure for $(name) of type $(typeof(v))" begin
                        @test hasfield(RT, :base)
                    end
                    if hasfield(RT, :base)
                        _testgrads(base(v), seen, res.base, exp, name)
                    end
                end
            end
            function _testgrads(v::CompVertex, seen, res, exp, name) 
                @testset "Check grads for $name" begin
                    # Ops like + and cat seem to get a gradient for contents even though they don't have such a field
                    if !isa(getindex(exp, name), NamedTuple{(:contents,)})
                        @test res.computation == getindex(exp, name)
                    end
                end
                foreach(enumerate(inputs(v))) do (i, vi)
                    isnothing(res.inputs) && return
                    testgrads(vi, seen, res.inputs[i], exp)
                end            
            end

            function makegraphs() 
                l1 = ImMatMul(2, 3)
                l2 = ImMatMul(3, 3)
                l4 = ImMatMul(6, 3)
                
                # Make CompGraph
                vi = inputvertex("in", 2)
                v1 = absorbvertex("l1", l1, vi)
                v2 = absorbvertex("l2", l2, v1)
                v3 = conc("v3", v1, v2; dims=1)
                v4 = absorbvertex("l4", l4, v3) 
                v5 = "v5" >> v1 + v4
                v6 = conc("v6", v4, v5; dims=1)
                graph = CompGraph(vi, v6)
              
                # Same function as graph but as a normal function
                graph, function fgraph(vi)
                    v1 = l1(vi)
                    v2 = l2(v1)
                    v3 = cat(v1, v2; dims=1)
                    v4 = l4(v3) 
                    v5 = v1 .+ v4
                    v6 = cat(v4, v5; dims=1)
                end          
            end

            x = reshape(collect(Float32, 1:6), 2, 3)
            @testset "Explicit gradients" begin
                graph, fgraph  = makegraphs()
                @test graph(x) == fgraph(x)

                @test gradient(sum ∘ graph, x) == gradient(sum ∘ fgraph, x)
                res = gradient(g -> sum(g(x)), graph)
                exp = gradient(f -> sum(f(x)), fgraph)
                testgrads(graph, res..., exp...)       
            end

            @testset "Implicit gradients" begin
                graph, fgraph  = makegraphs()
                @test graph(x) == fgraph(x)

                ps = getfield.(filter(c -> c isa ImMatMul, computation.(vertices(graph))), :W) |> Zygote.Params
                res = gradient(() -> sum(graph(x)), ps)
                exp = gradient(() -> sum(fgraph(x)), ps)

                @test length(ps) == length(res) == length(exp) == 3

                @testset "Gradient for $(name(v))" for v in filter(v -> computation(v) isa ImMatMul, vertices(graph))
                    p = computation(v).W
                    @test p in keys(res)
                    @test p in keys(exp)
                    @test res[p] == exp[p]
                end
            end
        end
    end
end