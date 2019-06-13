
@testset "Size mutations" begin

    inpt(size, id=1) = InputSizeVertex(id, size)

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

            Δnin(tv, +1)
            @test nout(iv) == nin(tv)[1] == nout(tv) == nin(io) == 4

            Δnout(tv, -1)
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

            Δnin(tv, -1, missing)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 5

            Δnin(tv, missing, 2)
            @test nout(iv1) + nout(iv2) == sum(nin(tv)) == nout(tv) == nin(io1) == nin(io2) == 7

            Δnout(tv, -1)
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
        nt(name) = t -> NamedTrait(t, name)
        rb(start, residual,name="add") = InvariantVertex(CompVertex(+, residual, start), nt(name))
        merge(paths...;name="merge") = StackingVertex(CompVertex(hcat, paths...), nt(name))
        mm(nin, nout) = x -> x * reshape(collect(1:nin*nout), nin, nout)
        av(op, state, in...;name="comp") = AbsorbVertex(CompVertex(op, in...), state, nt(name))
        function stack(start, nouts...; bname = "stack")
            # Can be done on one line with mapfoldl, but it is not pretty...
            next = start
            for i in 1:length(nouts)
                op = mm(nout(next), nouts[i])
                next = av(op, IoSize(nout(next), nouts[i]), next, name="$(bname)_$i")
            end
            return next
        end

        @testset "Residual fork block" begin
            start = av(mm(3,9), IoSize(3,9), InputSizeVertex("in", 3), name="start")
            p1 = stack(start, 3,4, bname = "p1")
            p2 = stack(start, 4,5, bname = "p2")
            resout = rb(start, merge(p1, p2))
            out = av(mm(9, 4), IoSize(9,4), resout; name="out")

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
            start = av(mm(3,8), IoSize(3,8), InputSizeVertex("in", 3), name="start")
            split = av(mm(8,4), IoSize(8,4), start, name="split")
            p1 = merge(split, name="p1") #Just an identity vertex
            p2 = stack(split, 3,2,4, bname="p2")
            resout = rb(start, merge(p1, p2))
            out = av(mm(8, 3), IoSize(8,3), resout, name="out")

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
            start = av(mm(3,4), IoSize(3,4), InputSizeVertex("in", 3), name="start")
            p1 = merge(start, name="p1")
            p2 = merge(start, name="p2")
            join = merge(p1, p2)
            out = av(mm(8, 3), IoSize(8,3), join, name="out")

            @test nout(join) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the merge before resout. Simple dfs will
            # fail as one will hit the merge through p1 before having
            # resolved the path through p2.
            Δnout(start, -1)
            @test nin(out) == [2nout(start)] == [6]

            # Should basically undo the previous mutation
            @test minΔnoutfactor_only_for.(inputs(out)) == [2]
            Δnin(out, +2)
            @test nin(out) == [2nout(start)] == [8]
        end

        @testset "Transparent residual fork block" begin
            start = av(mm(3,8), IoSize(3,8), InputSizeVertex("in", 3), name="start")
            split = av(mm(8,4), IoSize(8,4), start, name="split")
            p1 = merge(split, name="p1")
            p2 = merge(split, name="p2")
            resout = rb(start, merge(p1, p2))
            out = av(mm(8, 3), IoSize(8,3), resout, name="out")

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

            @test minΔnoutfactor_only_for.(inputs(out)) == [2]
            Δnout(start, -2)
            @test nin(out) == [nout(start)] == [6]
            @test nout(split) == 3
        end
    end

    @testset "Size Mutation possibilities" begin

        # Helpers
        struct SizeConstraint constraint; end
        NaiveNASlib.minΔnoutfactor(c::SizeConstraint) = c.constraint
        NaiveNASlib.minΔninfactor(c::SizeConstraint) = c.constraint
        av(size, csize, in...;name = "av") = AbsorbVertex(CompVertex(SizeConstraint(csize), in...), IoSize(vcat(nout.(in)...), size), t -> NamedTrait(t, name))
        sv(in...; name="sv") = StackingVertex(CompVertex(hcat, in...), t -> NamedTrait(t, name))
        iv(in...; name="iv") = InvariantVertex(CompVertex(hcat, in...), t -> NamedTrait(t, name))

        @testset "InputSizeVertex" begin
            @test ismissing(minΔnoutfactor(inpt(3)))
        end

        @testset "AbsorbVertex" begin
            v1 = av(4, 2, inpt(3))
            v2 = av(3, 1, v1)
            v3 = av(5, 3, v2)
            @test minΔnoutfactor(v1) == 2
            @test ismissing(minΔninfactor(v1))
            @test minΔnoutfactor(v3) == minΔninfactor(v3) == 3
            @test minΔnoutfactor(v2) == 3
            @test minΔninfactor(v2) == 2
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
            v1 = av(6,3, inpt(3), name="v1")
            v2 = av(7,2, inpt(3), name="v2")
            v3 = av(8,2, inpt(3), name="v3")

            sv1 = sv(v1, v2)
            sv2 = sv(v3, v2, v1, v2)
            @test minΔnoutfactor(sv1) == 6
            @test minΔnoutfactor(sv2) == 12

            # Expect only v1 to change as size change is not compatible with v2
            Δnout(sv1, -3)
            @test nout(v1) == 3
            @test nout(v2) == 7

            # v1 can't change as it is too small already
            # v3 makes a larger change as it is larger than v1
            Δnout(sv2, -6)
            @test nout(v1) == 3
            @test nout(v2) == 7
            @test nout(v3) == 2

            Δnout(sv2, +9)
            @test nout(v1) == 6
            @test nout(v2) == 9
            @test nout(v3) == 4
        end

        @testset "Stacked StackingVertices" begin
            v1 = av(100,1, inpt(3), name="v1")
            v2 = av(100,2, inpt(3), name="v2")
            v3 = av(100,3, inpt(3), name="v3")
            v4 = av(100,5, inpt(3), name="v4")

            sv1  = sv(v2, v3, name="sv1")
            sv2 = sv(sv1, v1, name="sv2")
            sv3 = sv(sv2, v4, v2, name="sv3")
            @test minΔnoutfactor(sv1) == minΔninfactor(sv1) == 6
            @test minΔnoutfactor(sv2) == minΔninfactor(sv2)== 6
            @test minΔnoutfactor(sv3) == minΔninfactor(sv3) == 60 # v2 is input twice through sc2->sv1
            Δnout(sv3, -60)
            @test nout(v1) == 86
            @test nout(v2) == 92
            @test nout(v3) == 85
            @test nout(v4) == 85
            @test nout(sv1) == nout(v2) + nout(v3) == 177
            @test nout(sv2) == nout(sv1) + nout(v1) == 263
            @test nout(sv3) == nout(sv2) + nout(v4) + nout(v2) == 440

            v5 = av(10, 3, sv3)
            @test minΔnoutfactor(sv1) == minΔninfactor(sv1) == 6
            @test minΔnoutfactor(sv2) == minΔninfactor(sv2)== 6
            @test minΔnoutfactor(sv3) == minΔninfactor(sv3) == 60

            Δnout(v1, 3)
            @test nout(v1) == 89
            @test nout(v2) == 92
            @test nout(v3) == 85
            @test nout(v4) == 85
            @test nout(sv1) == nout(v2) + nout(v3) == 177
            @test nout(sv2) == nout(sv1) + nout(v1) == 266
            @test [nout(sv3)] == [(nout(sv2) + nout(v4) + nout(v2))] == nin(v5) == [443]

            # Evil action! Must have understanding that the change in v2 will propagate
            # to sv3 input 1 through sv1 and sv2 and hold off updating it through input 3
            Δnout(v2, -13)
            @test nout(v1) == 89
            @test nout(v2) == 79
            @test nout(v3) == 85
            @test nout(v4) == 85
            @test nout(sv1) == nout(v2) + nout(v3) == 164
            @test nout(sv2) == nout(sv1) + nout(v1) == 253
            @test [nout(sv3)] == [(nout(sv2) + nout(v4) + nout(v2))] == nin(v5) == [417]

        end
        @testset "Stacked InvariantVertex" begin
            v1 = av(100,1, inpt(3), name="v1")
            v2 = av(100,2, inpt(3), name="v2")
            v3 = av(100,3, inpt(3), name="v3")
            v4 = av(100,5, inpt(3), name="v4")

            iv1 = iv(v2,v3, name="iv1")
            iv2 = iv(iv1, v1, name="iv2")
            iv3 = iv(iv2, v4, v2, name="iv2")

            @test minΔnoutfactor(iv1) == minΔninfactor(iv1) == 6
            @test minΔnoutfactor(iv2) == minΔninfactor(iv2) == 6
            @test minΔnoutfactor(iv3) == minΔninfactor(iv3) == 30


            Δnout(iv3, -30)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 70
            @test nout(iv1) == nout(iv2) == nout(iv3) == 70

            v5 = av(10, 3, iv3)

            Δnout(v1, 3)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 73
            @test [nout(iv1)] == [nout(iv2)] == [nout(iv3)] == nin(v5) == [73]

            # Evil action! Must have understanding that the change in v2 will propagate
            # to iv3 input 1 through iv1 and iv2 and hold off updating it through input 3
            Δnout(v2, -12)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 61
            @test [nout(iv1)] == [nout(iv2)] == [nout(iv3)] == nin(v5) == [61]
        end
    end

end
