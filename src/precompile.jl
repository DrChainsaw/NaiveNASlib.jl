let
    @setup_workload begin

        mutable struct MatMul{M<:AbstractMatrix}
            W::M
        end
        MatMul(nin, nout) = MatMul(reshape(collect(1:nin*nout), nout,nin))
        
        @functor MatMul
        
        (mm::MatMul)(x) = mm.W * x
        
        function NaiveNASlib.Δsize!(mm::MatMul, ins::AbstractVector, outs::AbstractVector)
            mm.W = NaiveNASlib.parselect(mm.W, 1=>outs, 2=>ins[1])
            nothing
        end
        NaiveNASlib.nout(mm::MatMul) = size(mm.W, 1)
        NaiveNASlib.nin(mm::MatMul) = [size(mm.W, 2)]
        
        x1 = ones(Float32, 1, 1)
        x2 = ones(Float32, 2, 1)

        l1 = MatMul(2, 3)
        l2 = MatMul(3, 3)
        l4 = MatMul(6, 3)


        @compile_workload begin

            let
                vi1,vi2 = inputvertex.("in", (1,1))
                v1 = "add" >> vi1 + vi2
                v2 = "mul" >> vi1 * v1
                v3 = "div" >> v1 / v2
                v4 = "sub" >> v3 - v2
                CompGraph([vi1, vi2], v4)(x1, x1)
                CompGraph([vi1, vi2], [v2, v4])(x1, x1)
            end

            let 
                vi = inputvertex("in", 2)
                v1 = absorbvertex("l1", l1, vi)
                v2 = absorbvertex("l2", l2, v1)
                v3 = conc("v3", v1, v2; dims=1)
                v4 = absorbvertex("l4", l4, v3) 
                v5 = "v5" >> v1 + v4
                graph = CompGraph(vi, v5)

                graph(x2)

                Δnout!(v1 => 1)
                Δnout!(v1 => 2, v2 => relaxed(-1))
                Δsize!(graph)
                create_edge!(v1, v4)
                create_edge!(v4, v5)
                remove_edge!(v1, v4)
                remove_edge!(v4, v5)
                remove!(v2)
                insert!(v1, v -> absorbvertex("l2new", l2, v1))
            end
        end
    end
end