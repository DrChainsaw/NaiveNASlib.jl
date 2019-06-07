
@testset "Size mutations" begin

    inpt(size, id=1) = InputSizeVertex(InputVertex(id), size)

    @testset "AbsorbVertex" begin
        iv = AbsorbVertex(InputVertex(1), InvSize(2))
        v1 = AbsorbVertex(CompVertex(x -> 3 .* x, iv), IoSize(2, 3))

        @test nin(v1) == [2]
        @test nout(v1) == 3

        @test outputs.(inputs(v1)) == [[v1]]

        Δnin(v1, 2)
        @test nin(v1) == [4]

        Δnin(v1, -3)
        @test nin(v1) == [1]

        Δnout(v1, -2)
        @test nout(v1) == 1

        # Add one vertex and see that change propagates
        v2 = AbsorbVertex(CompVertex(x -> 3 .+ x, v1), IoSize(1, 4)) # Note: Size after change above
        @test nout(v1) == nin(v2)[1]
        @test outputs(v1) == [v2]
        @test inputs(v2) == [v1]

        Δnin(v2, 4)
        @test nout(v1) == nin(v2)[1] == 5

        Δnout(v1, -2)
        @test nout(v1) == nin(v2)[1] == 3

        # Fork of v1 into a new vertex
        v3 = AbsorbVertex(CompVertex(identity, v1), IoSize(3, 2))
        @test outputs(v1) == [v2, v3]
        @test inputs(v3) == inputs(v2) == [v1]

        Δnout(v1, -2)
        @test nout(v1) == nin(v2)[1] == nin(v3)[1] == 1

        Δnin(v3, 3)
        @test nout(v1) == nin(v2)[1] == nin(v3)[1] == 4

        Δnin(v2, -2)
        @test nout(v1) == nin(v2)[1] == nin(v3)[1] == 2

        ivc = clone(iv)
        v1c = clone(v1, ivc)
        v2c = clone(v2, v1c)
        v3c = clone(v3, v1c)

        @test issame(iv, ivc)
        @test issame(v1, v1c)
        @test issame(v2, v2c)
        @test issame(v3, v3c)
    end

    @testset "StackingVertex" begin

        @testset "StackingVertex 1 to 1" begin

            iv = AbsorbVertex(CompVertex(identity, inpt(2)), InvSize(2))
            tv = StackingVertex(CompVertex(identity, iv))
            io = AbsorbVertex(CompVertex(identity, tv), InvSize(2))

            @test outputs.(inputs(io)) == [[io]]
            @test outputs.(inputs(tv)) == [[tv]]
            @test outputs(iv) == [tv]

            Δnout(iv, 3)
            @test nout(iv) == nin(tv)[1] == nout(tv) == nin(io) == 5

            Δnin(io, -2)
            @test nout(iv) == nin(tv)[1] == nout(tv) == nin(io) == 3

            ivc = clone(iv, inpt(2))
            tvc = clone(tv, ivc)
            ioc = clone(io, tvc)

            @test issame(iv, ivc)
            @test issame(tv, tvc)
            @test issame(io, ioc)
        end

        @testset "StackingVertex 2 inputs" begin
            # Try with two inputs to StackingVertex
            iv1 = AbsorbVertex(CompVertex(identity, inpt(2)), InvSize(2))
            iv2 = AbsorbVertex(CompVertex(identity, inpt(3,2)), InvSize(3))
            tv = StackingVertex(CompVertex(hcat, iv1, iv2))
            io1 = AbsorbVertex(CompVertex(identity, tv), InvSize(5))

            @test inputs(tv) == [iv1, iv2]
            @test outputs.(inputs(tv)) == [[tv], [tv]]
            @test outputs(iv1) == [tv]
            @test outputs(iv2) == [tv]

            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == 5

            Δnout(iv2, -2)
            @test nin(io1) == nout(tv) == sum(nin(tv)) == nout(iv1) + nout(iv2)  ==  3

            Δnin(io1, 3)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == 6

            Δnout(iv1, 4)
            Δnin(io1, -8)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == 2
            @test nout(iv1) == nout(iv2) == 1

            #Add another output
            io2 = AbsorbVertex(CompVertex(identity, tv), InvSize(2))

            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 2

            Δnout(iv1, 3)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 5

            Δnin(io2, -2)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 3

            Δnin(io1, 3)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 6

            iv1c = clone(iv1, inpt(2))
            iv2c = clone(iv2, inpt(3,2))
            tvc = clone(tv, iv1c, iv2c)
            io1c = clone(io1, tvc)
            io2c = clone(io2, tvc)

            @test issame(iv1, iv1c)
            @test issame(iv2, iv2c)
            @test issame(tv, tvc)
            @test issame(io1, io1c)
            @test issame(io2, io2c)
        end
    end
    @testset "InvariantVertex" begin

        @testset "InvariantVertex 1 to 1" begin
            #Behaviour is identical to StackingVertex in this case
            iv = AbsorbVertex(InputVertex(1), InvSize(2))
            inv = InvariantVertex(CompVertex(identity, iv))
            io = AbsorbVertex(CompVertex(identity, inv), InvSize(2))

            @test outputs.(inputs(io)) == [[io]]
            @test outputs.(inputs(inv)) == [[inv]]
            @test outputs(iv) == [inv]

            Δnout(iv, 3)
            @test nout(iv) == nin(inv)[1] == nout(inv) == nin(io) == 5

            Δnin(io, -2)
            @test nout(iv) == nin(inv)[1] == nout(inv) == nin(io) == 3

            ivc = clone(iv)
            invc = clone(inv, ivc)
            ioc = clone(io, invc)

            @test issame(iv, ivc)
            @test issame(inv, invc)
            @test issame(io, ioc)

        end

        @testset "InvariantVertex 2 inputs" begin
            # Try with two inputs a InvariantVertex
            iv1 = AbsorbVertex(InputVertex(1), InvSize(3))
            iv2 = AbsorbVertex(InputVertex(2), InvSize(3))
            inv = InvariantVertex(CompVertex(+, iv1, iv2))
            io1 = AbsorbVertex(CompVertex(identity, inv), InvSize(3))

            @test inputs(inv) == [iv1, iv2]
            @test outputs.(inputs(inv)) == [[inv], [inv]]
            @test outputs(iv1) == [inv]
            @test outputs(iv2) == [inv]

            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == 3

            Δnout(iv2, -2)
            @test nin(io1) == nout(inv) == nin(inv)[1] == nin(inv)[2]  == nout(iv1) == nout(iv2)  ==  1

            Δnin(io1, 3)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == 4

            #Add another output
            io2 = AbsorbVertex(CompVertex(identity, inv), InvSize(4))

            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == nin(io2) == 4

            Δnout(iv1, -3)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == nin(io2) == 1

            Δnin(io2, 2)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == nin(io2) == 3

            Δnout(iv2, 3)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1) == nin(io2) == 6

            iv1c = clone(iv1)
            iv2c = clone(iv2)
            invc = clone(inv, iv1c, iv2c)
            io1c = clone(io1, invc)
            io2c = clone(io2, invc)

            @test issame(iv1, iv1c)
            @test issame(iv2, iv2c)
            @test issame(inv, invc)
            @test issame(io1, io1c)
            @test issame(io2, io2c)

        end

    end

    @testset "Mutate tricky structures" begin

        ## Helper functions
        rb(start, residual) = InvariantVertex(CompVertex(+, residual, start))
        merge(paths...) = StackingVertex(CompVertex(hcat, paths...))
        mm(nin, nout) = x -> x * reshape(collect(1:nin*nout), nin, nout)
        av(op, state, in...) = AbsorbVertex(CompVertex(op, in...), state)
        function stack(start, nouts...)
            # Can be done on one line with mapfoldl, but it is not pretty...
            next = start
            for i in 1:length(nouts)
                op = mm(nout(next), nouts[i])
                next = av(op, IoSize(nout(next), nouts[i]), next)
            end
            return next
        end

        @testset "Residual fork block" begin
            start = AbsorbVertex(InputVertex(1), InvSize(9))
            p1 = stack(start, 3,4)
            p2 = stack(start, 4,5)
            resout = rb(start, merge(p1, p2))
            out = av(mm(9, 4), IoSize(9,4), resout)

            @test nout(resout) == 9

            # Propagates to out, start outputs of start
            Δnout(p2, -2)
            @test nin(out) == [nout(start)] == [7]
            #outputs(start) = first vertex in p1, p2 and resout (which has two inputs)
            @test foldl(vcat, nin.(outputs(start))) == [7, 7, 7, 7]

            # Should basically undo the previous mutation
            Δnin(out, +2)
            @test nin(out) == [nout(start)] == [9]
            @test foldl(vcat, nin.(outputs(start))) == [9, 9, 9, 9]
        end

        @testset "Half transparent residual fork block" begin
            start = AbsorbVertex(InputVertex(1), InvSize(8))
            split = av(mm(8,4), IoSize(8,4), start)
            p1 = StackingVertex(CompVertex(identity, split))
            p2 = stack(split, 3,2,4)
            resout = rb(start, merge(p1, p2))
            out = av(mm(8, 3), IoSize(8,3), resout)

            @test nout(resout) == 8

            # Propagates to input of first vertex of p2, input of out and start
            # via p1 and resout as well as to input of split
            Δnout(split, -1)
            @test nin(out) == [nout(start)] == nin(split) == [7]
            @test foldl(vcat, nin.(outputs(split))) == [3, 3]

            # Should basically undo the previous mutation
            Δnin(out, +1)
            @test nin(out) == [nout(start)] == nin(split) == [8]
        end

        @testset "Transparent fork block" begin
            start = AbsorbVertex(CompVertex(identity, inpt(4)), InvSize(4))
            p1 = StackingVertex(CompVertex(identity, start))
            p2 = StackingVertex(CompVertex(identity, start))
            join = merge(p1, p2)
            out = av(mm(8, 3), IoSize(8,3), join)

            @test nout(join) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the merge before resout. Simple dfs will
            # fail as one will hit the merge through p1 before having
            # resolved the path through p2.
            Δnout(start, -1)
            @test nin(out) == [2nout(start)] == [6]

            # Should basically undo the previous mutation
            @test minΔnoutfactor.(inputs(out)) == [2]
            Δnin(out, +2)
            @test nin(out) == [2nout(start)] == [8]
        end

        @testset "Transparent residual fork block" begin
            start = AbsorbVertex(CompVertex(identity, inpt(8)), InvSize(8))
            split = av(mm(8,4), IoSize(8,4), start)
            p1 = StackingVertex(CompVertex(identity, split))
            p2 = StackingVertex(CompVertex(identity, split))
            resout = rb(start, merge(p1, p2))
            out = av(mm(8, 3), IoSize(8,3), resout)

            @test nout(resout) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the merge before resout. Simple dfs will
            # fail as one will hit the merge through p1 before having
            # resolved the path through p2.
            Δnout(split, -1)
            @test nin(out) == [nout(start)] == nin(split) == [6]

            # Should basically undo the previous mutation
            Δnin(out, +2)
            @test nin(out) == [nout(start)] == nin(split) == [8]

            @test minΔnoutfactor.(inputs(out)) == [2]
            Δnout(start, -4)
            @test nin(out) == [nout(start)] == [4]
            @test nout(split) == 2
        end
    end

    @testset "Size Mutation possibilities" begin

        # Helpers
        struct SizeConstraint constraint; end
        NaiveNASlib.minΔnoutfactor(c::SizeConstraint) = c.constraint
        av(size, csize, in...) = AbsorbVertex(CompVertex(SizeConstraint(csize), in...), IoSize(collect(nin.(in)), size))
        sv(in...) = StackingVertex(CompVertex(hcat, in...))

        @testset "InputSizeVertex" begin
            @test ismissing(minΔnoutfactor(inpt(3)))
        end

        @testset "AbsorbVertex" begin
            @test minΔnoutfactor(av(4, 2, inpt(3))) == 2
        end

        @testset "Pick values" begin
            import NaiveNASlib: pick_values

            @test pick_values([1], 1) == [1]
            @test pick_values([17], 17) == [1]
            @test pick_values([3,2], 12) == [2, 3]
            @test pick_values([3,2,1], 12) == [2,2,2]

            @test pick_values([5,3,1], 37) == [4,4,5]
            @test pick_values([5,3,1], 37, sum) == [5,4,0]

            vals = [11, 7, 5, 3, 1]
            for target in 1:10
                @test sum(pick_values(vals, target) .* vals) == target
            end
        end

        @testset "StackingVertex single input" begin
            @test ismissing(minΔnoutfactor(sv(inpt(3))))
            @test minΔnoutfactor(sv(av(4, 2, inpt(3)))) == 2
        end

        @testset "StackingVertex multi inputs" begin
            v1 = av(5,3, inpt(3))
            v2 = av(6,2, inpt(3))
            v3 = av(8,2, inpt(3))


            sv1 = sv(v1, v2)
            sv2 = sv(v3, v2, v1, v2)
            @test minΔnoutfactor(sv1) == 3
            @test minΔnoutfactor(sv2) == 4

            # Expect only v1 to change as size change is not compatible with v2
            Δnout(sv1, -3)
            @test nout(v1) == 2
            @test nout(v2) == 6

            # v1 can't change as it is too small already
            # v3 makes a larger change as it is larger than v1
            Δnout(sv2, -6)
            @test nout(v1) == 2
            @test nout(v2) == 4
            @test nout(v3) == 6

            Δnout(sv2, +9)
            @test nout(v1) == 5
            @test nout(v2) == 6
            @test nout(v3) == 8

        end

    end

end
