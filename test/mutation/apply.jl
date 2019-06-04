using NaiveNASlib
import InteractiveUtils:subtypes

# Testing mock
mutable struct MatMul
    W::AbstractMatrix
end
MatMul(nin, nout) = MatMul(reshape(collect(1:nin*nout), nin,nout))
(mm::MatMul)(x) = x * mm.W
function NaiveNASlib.mutate_inputs(mm::MatMul, inputs::AbstractArray{<:Integer, 1}...)
    indskeep = filter(ind -> ind > 0, inputs[1])
    newmap = inputs[1] .> 0

    newmat = zeros(Int64, length(newmap), size(mm.W, 2))
    newmat[newmap, :] = mm.W[indskeep, :]
    mm.W = newmat
end
function NaiveNASlib.mutate_outputs(mm::MatMul, outputs::AbstractArray{<:Integer, 1})
    indskeep = filter(ind -> ind > 0, outputs)
    newmap = outputs .> 0

    newmat = zeros(Int64, size(mm.W, 1), length(newmap))
    newmat[:, newmap] = mm.W[:, indskeep]
    mm.W = newmat
end

@testset "Mutation testing" begin

    function mcv(nin, nout, in)
        mm = MatMul(nin, nout)
        return AbsorbVertex(CompVertex(mm, in), IoIndices(nin, nout)), mm
    end

    @testset "AbsorbVertex mutation" begin
        nin1, nout1 = 3, 5
        nin2, nout2 = 5, 2
        iv1 = AbsorbVertex(InputVertex(1), InvSize(nin1))
        mmv1, mm1 = mcv(nin1, nout1, iv1)
        mmv2, mm2 = mcv(nin2, nout2, mmv1)

        # Select subset
        Δnin(mmv2, Integer[1, 3, 4])
        apply_mutation.((mmv1, mmv2))
        @test mm1.W == [ 1 7 10; 2 8 11; 3 9 12]
        @test mm2.W == [ 1 6; 3 8; 4 9]

        # Increase size
        Δnout(mmv1, [collect(1:nout(mmv1))..., -1, -1])
        apply_mutation.((mmv1, mmv2))
        @test mm1.W == [ 1 7 10 0 0; 2 8 11 0 0; 3 9 12 0 0]
        @test mm2.W == [ 1 6; 3 8; 4 9; 0 0; 0 0]

        # Select subset and increase size
        Δnin(mmv2, Integer[1, -1, 3, -1, -1])
        apply_mutation.((mmv1, mmv2))
        @test mm1.W == [ 1 0 10 0 0; 2 0 11 0 0; 3 0 12 0 0]
        @test mm2.W == [ 1 6; 0 0; 4 9; 0 0; 0 0]
    end

    @testset "StackingVertex mutation" begin

        mmv1, mm1 = mcv(2, 3, NaiveNASlib.OutputsVertex(InputVertex(1)))
        mmv2, mm2 = mcv(3, 4, NaiveNASlib.OutputsVertex(InputVertex(2)))
        join = StackingVertex(CompVertex(hcat, mmv1, mmv2))
        mmv3, mm3 = mcv(7, 3, join)

        Δnout(join, Integer[1,3,5,7])
        apply_mutation.((mmv1, mmv2, mmv3))

        @test mm1.W == [1 5; 2 6]
        @test mm2.W == [4 10; 5 11; 6 12]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 7 14 21]

        Δnout(mmv2, [1, -1, -1])
        apply_mutation.((mmv1, mmv2, mmv3))

        @test mm1.W == [1 5; 2 6]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 0 0 0; 0 0 0]

        Δnout(mmv1, [2, -1])
        apply_mutation.((mmv1, mmv2, mmv3))

        @test mm1.W == [5 0; 6 0]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [3 10 17; 0 0 0; 5 12 19; 0 0 0; 0 0 0]

        Δnin(mmv3, [1, 3])
        apply_mutation.((mmv1, mmv2, mmv3))

        @test mm1.W == reshape([5; 6], :, 1)
        @test mm2.W == reshape([4; 5; 6], :, 1)
        @test mm3.W == [3 10 17; 5 12 19]

        Δnin(mmv3, [1, -1, 2, -1, -1])
        apply_mutation.((mmv1, mmv2, mmv3))

        @test mm1.W == [5 0 ; 6 0]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [3 10 17; 0 0 0; 5 12 19; 0 0 0; 0 0 0]
    end

    @testset "InvariantVertex mutation" begin

        addsize = 7
        mmv1, mm1 = mcv(2, addsize, NaiveNASlib.OutputsVertex(InputVertex(1)))
        mmv2, mm2 = mcv(3, addsize, NaiveNASlib.OutputsVertex(InputVertex(2)))
        add = InvariantVertex(CompVertex(+, mmv1, mmv2))
        mmv3, mm3 = mcv(addsize, 3, add)

        Δnout(add, Integer[1,3,5,7])
        apply_mutation.((mmv1, mmv2, mmv3))

        @test mm1.W == [1 5 9 13; 2 6 10 14]
        @test mm2.W == [1 7 13 19; 2 8 14 20; 3 9 15 21]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 7 14 21]

        Δnout(mmv2, [2, -1, 3])
        apply_mutation.((mmv1, mmv2, mmv3))

        @test mm1.W == [5 0 9; 6 0 10]
        @test mm2.W == [7 0 13; 8 0 14; 9 0 15]
        @test mm3.W == [3 10 17; 0 0 0; 5 12 19]

        Δnout(mmv1, [-1, 1, 2, -1])
        apply_mutation.((mmv1, mmv2, mmv3))

        @test mm1.W == [0 5 0 0; 0 6 0 0]
        @test mm2.W == [0 7 0 0; 0 8 0 0; 0 9 0 0]
        @test mm3.W == [0 0 0; 3 10 17; 0 0 0; 0 0 0]

        Δnin(mmv3, [2])
        apply_mutation.((mmv1, mmv2, mmv3))

        # Need to add one dim: 2 -> 2x1
        @test mm1.W == reshape([5; 6], :, 1)
        @test mm2.W == reshape([7; 8; 9], :, 1)
        @test mm3.W == [3 10 17]

        Δnin(mmv3, [-1, 1, -1])
        apply_mutation.((mmv1, mmv2, mmv3))

        @test mm1.W == [0 5 0; 0 6 0]
        @test mm2.W == [0 7 0; 0 8 0; 0 9 0]
        @test mm3.W == [0 0 0; 3 10 17; 0 0 0]
    end

    @testset "InvariantVertex with mutation OP" begin
        invsize = 3
        iv, mmi = mcv(2, invsize, NaiveNASlib.OutputsVertex(InputVertex(1)))
        mminv = MatMul(invsize, invsize)
        inv = InvariantVertex(CompVertex(mminv, iv), InvIndices(invsize))

        Δnout(inv, [1, 3])
        apply_mutation.((inv, iv))

        @test mminv.W == [1 7; 3 9]
        @test mmi.W == [1 5; 2 6]

        Δnin(inv, [-1, 2, -1])
        apply_mutation.((inv, iv))

        @test mminv.W == [0 0 0;0 9 0; 0 0 0]
        @test mmi.W == [0 5 0; 0 6 0]
    end

    @testset "Size only mutation" begin
        mutable struct SizeProbe
            nin
            nout
        end
        NaiveNASlib.mutate_inputs(p::SizeProbe, newsize...) = p.nin = newsize[1]
        NaiveNASlib.mutate_outputs(p::SizeProbe, newsize) = p.nout = newsize

        @testset "IoSize" begin
            p = SizeProbe(3,5)
            in = AbsorbVertex(CompVertex(identity, NaiveNASlib.OutputsVertex(InputVertex(1))), IoSize(1, 3))
            v = AbsorbVertex(CompVertex(p, in), IoSize(p.nin, p.nout))

            Δnin(v, -1)
            apply_mutation(v)
            @test p.nin == 2
            @test p.nout == 5

            Δnout(v, 2)
            apply_mutation(v)
            @test p.nin == 2
            @test p.nout == 7
        end

        @testset "InvSize" begin
            p = SizeProbe(5,5)
            in = AbsorbVertex(CompVertex(identity, NaiveNASlib.OutputsVertex(InputVertex(1))), IoSize(1, 5))
            v = AbsorbVertex(CompVertex(p, in), InvSize(p.nin))

            Δnin(v, -1)
            apply_mutation(v)
            @test p.nin == 4
            @test p.nout == 4

            Δnout(v, 2)
            apply_mutation(v)
            @test p.nin == 6
            @test p.nout == 6
        end

    end

    @testset "Mutate tricky structures" begin

        ## Helper functions
        rb(start, residual) = InvariantVertex(CompVertex(+, residual, start))
        merge(paths...) = StackingVertex(CompVertex(hcat, paths...))
        function stack(start, nouts...)
            next = start
            mm = missing
            for i in 1:length(nouts)
                next, mm = mcv(nout(next), nouts[i], next)
            end
            return next, mm
        end

        @testset "Residual fork block" begin
            start, mmstart = mcv(2, 9, NaiveNASlib.OutputsVertex(InputVertex(1)))
            p1, mmp1 = stack(start, 3,4)
            p2, mmp2 = stack(start, 4,5)
            resout = rb(start, merge(p1, p2))
            out, mmout = mcv(9, 3, resout)

            @test nout(resout) == 9

            # Propagates to out, start outputs of start
            Δnout(p2, [2, 4])
            apply_mutation.(flatten(out))

            @test mmp1.W == [1 4 7 10; 2 5 8 11; 3 6 9 12] #Not touched
            @test mmp2.W == [5 13; 6 14; 7 15; 8 16] #Columns 2 and 4 kept
            @test mmstart.W == [1 3 5 7 11 15; 2 4 6 8 12 16] #Columns 1:4 6 and 8 kept
            @test mmout.W == [1 10 19; 2 11 20; 3 12 21; 4 13 22; 6 15 24; 8 17 26] # Rows 1:4, 6 and 8 kept

            Δnin(out, [2, 3, -1, 5, -1])
            apply_mutation.(flatten(out))

            @test mmp1.W == [4 7 0; 5 8 0; 6 9 0] #Columns 2 and 3 kept, new column added to end
            @test mmp2.W == [5 0; 6 0; 7 0; 8 0] #Column 1 kept, new column added to end
            @test mmstart.W == [3 5 0 11 0; 4 6 0 12 0] #Columns 2,3 5 kept, two new columns
            @test mmout.W == [2 11 20; 3 12 21; 0 0 0; 6 15 24; 0 0 0] # Rows 2,3 5 kept, two new rows
        end

        @testset "Half transparent residual fork block" begin
            start, mmstart = mcv(2, 5, NaiveNASlib.OutputsVertex(InputVertex(1)))
            split, mmsplit = mcv(5, 3, start)
            p1 = StackingVertex(CompVertex(identity, split))
            p2,mmp2 = stack(split, 3,2,2)
            resout = rb(start, merge(p1, p2))
            out, mmout = mcv(5, 3, resout)

            @test nout(resout) == 5

            # Propagates to input of first vertex of p2, input of out and start
            # via p1 and resout as well as to input of split
            Δnout(split, [1, 3])
            apply_mutation.(flatten(out))

            mmp2in = base(base(outputs(split)[2])).computation
            @test mmsplit.W == [1 11; 3 13; 4 14; 5 15]
            @test mmout.W == [1 6 11; 3 8 13; 4 9 14; 5 10 15]
            @test mmstart.W == [1 5 7 9; 2 6 8 10]
            @test mmp2in.W == [1 4 7; 3 6 9]

            Δnin(out, [2,-1, 3, 4, -1])
            apply_mutation.(flatten(out))

            @test mmsplit.W == [13 0; 0 0; 14 0; 15 0; 0 0]
            @test mmout.W == [3 8 13; 0 0 0; 4 9 14; 5 10 15; 0 0 0]
            @test mmstart.W == [5 0 7 9 0; 6 0 8 10 0]
            @test mmp2in.W == [3 6 9; 0 0 0]

        end

        @testset "Transparent fork block" begin
            start, mmstart = mcv(2, 4, NaiveNASlib.OutputsVertex(InputVertex(1)))
            p1 = StackingVertex(CompVertex(identity, start))
            p2 = StackingVertex(CompVertex(identity, start))
            join = merge(p1, p2)
            out,mmout = mcv(8, 3, join)

            @test nout(join) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the merge before resout. Simple dfs will
            # fail as one will hit the merge through p1 before having
            # resolved the path through p2.
            Δnout(start, [1,3,4])
            apply_mutation.(flatten(out))

            @test mmstart.W == [1 5 7; 2 6 8]
            @test mmout.W == [1 9 17; 3 11 19; 4 12 20; 5 13 21; 7 15 23; 8 16 24]

            Δnin(out, [-1, 2, -1, 3])
            apply_mutation.(flatten(out))

            @test mmstart.W == [0 5 0 7; 0 6 0 8]
            @test mmout.W == [0 0 0; 3 11 19; 0 0 0; 4 12 20]

        end

        @testset "Transparent residual fork block" begin
            start, mmstart = mcv(2, 8, NaiveNASlib.OutputsVertex(InputVertex(1)))
            split, mmsplit = mcv(8, 4, start)
            p1 = StackingVertex(CompVertex(identity, split))
            p2 = StackingVertex(CompVertex(identity, split))
            resout = rb(start, merge(p1, p2))
            out,mmout = mcv(8, 3, resout)

            @test nout(resout) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the merge before resout. Simple dfs will
            # fail as one will hit the merge through p1 before having
            # resolved the path through p2.
            Δnout(split, [2, 3, -1])
            apply_mutation.(flatten(out))

            @test mmsplit.W == [10 18 0; 11 19 0; 0 0 0; 14 22 0; 15 23 0; 0 0 0]
            @test mmout.W == [2 10 18; 3 11 19; 0 0 0; 6 14 22; 7 15 23; 0 0 0]
            @test mmstart.W == [3 5 0 11 13 0; 4 6 0 12 14 0]

            Δnin(out, [1, -1, 2, 3])
            apply_mutation.(flatten(out))

            @test mmsplit.W == [10 0 18 0; 0 0 0 0; 11 0 19 0; 0 0 0 0]
            @test mmout.W == [2 10 18; 0 0 0; 3 11 19; 0 0 0]
            @test mmstart.W == [3 0 5 0; 4 0 6 0]

            # Propagates to both inputs and outputs of split
            Δnout(start, [1, 3])
            apply_mutation.(flatten(out))

            @test mmsplit.W == [10 18; 11 19]
            @test mmout.W == [2 10 18; 3 11 19]
            @test mmstart.W == [3 5; 4 6]
        end

    end

end
