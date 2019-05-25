import NaiveNASlib
import InteractiveUtils:subtypes

using Test

# Testing mock
mutable struct MatMul
    W::AbstractMatrix
end
MatMul(nin, nout) = MatMul(reshape(collect(1:nin*nout), nin,nout))
(mm::MatMul)(x) = x * mm.W
function NaiveNASlib.select_inputs(mm::MatMul, inputs::AbstractArray{<:Integer, 1}...)
    indskeep = filter(ind -> ind <= size(mm.W, 1), inputs[1])
    toadd = length(inputs[1]) - length(indskeep)
    mm.W = mm.W[indskeep, :]
    mm.W = vcat(mm.W, zeros(Int64, toadd, size(mm.W, 2)))
end
function NaiveNASlib.select_outputs(mm::MatMul, outputs::AbstractArray{<:Integer, 1})
    indskeep =filter(ind -> ind <= size(mm.W, 2), outputs)
    toadd = length(outputs) - length(indskeep)
    mm.W = mm.W[:, indskeep]
    mm.W = hcat(mm.W, zeros(Int64, size(mm.W, 1), toadd))
 end


@testset "Selection testing" begin

    function mcv(nin, nout, in)
        mm = MatMul(nin, nout)
        return AbsorbVertex(CompVertex(mm, in), IoIndices(nin, nout)), mm
    end

    @testset "AbsorbVertex selection" begin
        nin1, nout1 = 3, 5
        nin2, nout2 = 5, 2
        iv1 = AbsorbVertex(InputVertex(1), InvSize(nin1))
        mmv1, mm1 = mcv(nin1, nout1, iv1)
        mmv2, mm2 = mcv(nin2, nout2, mmv1)

        # Select subset
        Δnin(mmv2, Integer[1, 3, 4])
        select_params(mmv1)
        select_params(mmv2)
        @test mm1.W == [ 1 7 10; 2 8 11; 3 9 12]
        @test mm2.W == [ 1 6; 3 8; 4 9]

        # Increase size
        Δnout(mmv1, collect(1:nout(mmv1) + 2))
        select_params(mmv1)
        select_params(mmv2)
        @test mm1.W == [ 1 7 10 0 0; 2 8 11 0 0; 3 9 12 0 0]
        @test mm2.W == [ 1 6; 3 8; 4 9; 0 0; 0 0]

        # Select subset and increase size
        Δnin(mmv2, Integer[1, 3, 4, 5, 6])
        select_params(mmv1)
        select_params(mmv2)
        @test mm1.W == [ 1 10 0 0 0; 2 11 0 0 0; 3 12 0 0 0]
        @test mm2.W == [ 1 6; 4 9; 0 0; 0 0; 0 0]
    end

end
