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
    toadd = -filter(ind -> ind < 0, inputs[1])
    newmap = Integer[]
    sizenew = length(inputs[1])
    toadd .-= minimum(abs.(indskeep)) .- 1

    foreach(y -> push!(newmap, setdiff(1:sizenew, toadd, newmap)[1]), indskeep)

    newmat = zeros(Int64, sizenew, size(mm.W, 2))
    newmat[newmap, :] = mm.W[indskeep, :]
    mm.W = newmat
end
function NaiveNASlib.select_outputs(mm::MatMul, outputs::AbstractArray{<:Integer, 1})
    indskeep = filter(ind -> ind > 0, outputs)
    toadd = -filter(ind -> ind < 0, outputs)
    newmap = Integer[]
    sizenew = length(outputs)

    foreach(y -> push!(newmap, setdiff(1:sizenew, toadd, newmap)[1]), indskeep)

    newmat = zeros(Int64, size(mm.W, 1), sizenew)
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
        Δnout(mmv1, [collect(1:nout(mmv1))..., -4, -5])
        select_params.((mmv1, mmv2))
        @test mm1.W == [ 1 7 10 0 0; 2 8 11 0 0; 3 9 12 0 0]
        @test mm2.W == [ 1 6; 3 8; 4 9; 0 0; 0 0]

        # Select subset and increase size
        Δnin(mmv2, Integer[1, 3, -4, -5, -6])
        select_params.((mmv1, mmv2))
        @test mm1.W == [ 1 10 0 0 0; 2 11 0 0 0; 3 12 0 0 0]
        @test mm2.W == [ 1 6; 4 9; 0 0; 0 0; 0 0]
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

        Δnout(mmv2, [1, -3, -4])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [1 5; 2 6]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 0 0 0; 0 0 0]

        Δnout(mmv1, [2, -3])
        select_params.((mmv1, mmv2, mmv3))

        @test mm1.W == [5 0; 6 0]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [3 10 17; 0 0 0; 5 12 19; 0 0 0; 0 0 0]


        #display(mm3.W)

    end

end
