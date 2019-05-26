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
    indskeep = filter(ind -> ind > 0, inputs[1])
    newmap = inputs[1] .> 0

    newmat = zeros(Int64, length(newmap), size(mm.W, 2))
    newmat[newmap, :] = mm.W[indskeep, :]
    mm.W = newmat
end
function NaiveNASlib.select_outputs(mm::MatMul, outputs::AbstractArray{<:Integer, 1})
    indskeep = filter(ind -> ind > 0, outputs)
    newmap = outputs .> 0

    newmat = zeros(Int64, size(mm.W, 1), length(newmap))
    newmat[:, newmap] = mm.W[:, indskeep]
    mm.W = newmat
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
        select_params.((mmv1, mmv2))
        @test mm1.W == [ 1 7 10; 2 8 11; 3 9 12]
        @test mm2.W == [ 1 6; 3 8; 4 9]

        # Increase size
        Δnout(mmv1, [collect(1:nout(mmv1))..., -1, -1])
        select_params.((mmv1, mmv2))
        @test mm1.W == [ 1 7 10 0 0; 2 8 11 0 0; 3 9 12 0 0]
        @test mm2.W == [ 1 6; 3 8; 4 9; 0 0; 0 0]

        # Select subset and increase size
        Δnin(mmv2, Integer[1, -1, 3, -1, -1])
        select_params.((mmv1, mmv2))
        @test mm1.W == [ 1 0 10 0 0; 2 0 11 0 0; 3 0 12 0 0]
        @test mm2.W == [ 1 6; 0 0; 4 9; 0 0; 0 0]
    end

    @testset "StackingVertex selection" begin

        mmv1, mm1 = mcv(2, 3, NaiveNASlib.OutputsVertex(InputVertex(1)))
        mmv2, mm2 = mcv(3, 4, NaiveNASlib.OutputsVertex(InputVertex(2)))
        join = StackingVertex(CompVertex(hcat, mmv1, mmv2))
        mmv3, mm3 = mcv(7, 3, join)

        Δnout(join, Integer[1,3,5,7])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [1 5; 2 6]
        @test mm2.W == [4 10; 5 11; 6 12]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 7 14 21]

        Δnout(mmv2, [1, -1, -1])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [1 5; 2 6]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 0 0 0; 0 0 0]

        Δnout(mmv1, [2, -1])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [5 0; 6 0]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [3 10 17; 0 0 0; 5 12 19; 0 0 0; 0 0 0]

        Δnin(mmv3, [1, 3])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == reshape([5; 6], :, 1)
        @test mm2.W == reshape([4; 5; 6], :, 1)
        @test mm3.W == [3 10 17; 5 12 19]

        Δnin(mmv3, [1, -1, 2, -1, -1])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [5 0 ; 6 0]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [3 10 17; 0 0 0; 5 12 19; 0 0 0; 0 0 0]
    end

    @testset "InvariantVertex selection" begin

        addsize = 7
        mmv1, mm1 = mcv(2, addsize, NaiveNASlib.OutputsVertex(InputVertex(1)))
        mmv2, mm2 = mcv(3, addsize, NaiveNASlib.OutputsVertex(InputVertex(2)))
        add = InvariantVertex(CompVertex(+, mmv1, mmv2))
        mmv3, mm3 = mcv(addsize, 3, add)

        Δnout(add, Integer[1,3,5,7])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [1 5 9 13; 2 6 10 14]
        @test mm2.W == [1 7 13 19; 2 8 14 20; 3 9 15 21]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 7 14 21]

        Δnout(mmv2, [2, -1, 3])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [5 0 9; 6 0 10]
        @test mm2.W == [7 0 13; 8 0 14; 9 0 15]
        @test mm3.W == [3 10 17; 0 0 0; 5 12 19]

        Δnout(mmv1, [-1, 1, 2, -1])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [0 5 0 0; 0 6 0 0]
        @test mm2.W == [0 7 0 0; 0 8 0 0; 0 9 0 0]
        @test mm3.W == [0 0 0; 3 10 17; 0 0 0; 0 0 0]

        Δnin(mmv3, [2])
        select_params.((mmv1, mmv2, mmv3))

        # Need to add one dim: 2 -> 2x1
        @test mm1.W == reshape([5; 6], :, 1)
        @test mm2.W == reshape([7; 8; 9], :, 1)
        @test mm3.W == [3 10 17]

        Δnin(mmv3, [-1, 1, -1])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [0 5 0; 0 6 0]
        @test mm2.W == [0 7 0; 0 8 0; 0 9 0]
        @test mm3.W == [0 0 0; 3 10 17; 0 0 0]
    end

end
