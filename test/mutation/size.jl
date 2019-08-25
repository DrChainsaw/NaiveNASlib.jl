@testset "Size mutations" begin
    using LightGraphs, MetaGraphs

    inpt(size, id=1) = InputSizeVertex(id, size)
    nt(name) = t -> NamedTrait(t, name)
    tf(name) = t -> nt(name)(SizeChangeValidation(t))

    @testset "AbsorbVertex" begin
        iv = AbsorbVertex(InputVertex(1), InvSize(2))
        v1 = AbsorbVertex(CompVertex(x -> 3 .* x, iv), IoSize(2, 3), tf("v1"))

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
        v2 = AbsorbVertex(CompVertex(x -> 3 .+ x, v1), IoSize(1, 4), tf("v2")) # Note: Size after change above
        @test nout(v1) == nin(v2)[1]
        @test outputs(v1) == [v2]
        @test inputs(v2) == [v1]

        Δnin(v2, 4)
        @test nout(v1) == nin(v2)[1] == 5

        Δnout(v1, -2)
        @test nout(v1) == nin(v2)[1] == 3

        # Fork of v1 into a new vertex
        v3 = AbsorbVertex(CompVertex(identity, v1), IoSize(3, 2), tf("v3"))
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
            @test [nout(iv)] == nin(tv) == [nout(tv)] == nin(io) == [5]

            Δnin(io, -2)
            @test [nout(iv)] == nin(tv) == [nout(tv)] == nin(io) == [3]

            Δnin(tv, +1)
            @test [nout(iv)] == nin(tv) == [nout(tv)] == nin(io) == [4]

            Δnout(tv, -1)
            @test [nout(iv)] == nin(tv) == [nout(tv)] == nin(io) == [3]

            ivc = clone(iv, inpt(2))
            tvc = clone(tv, ivc)
            ioc = clone(io, tvc)

            @test issame(iv, ivc)
            @test issame(tv, tvc)
            @test issame(io, ioc)
        end

        @testset "StackingVertex 2 inputs" begin
            # Try with two inputs to StackingVertex
            iv1 = AbsorbVertex(CompVertex(identity, inpt(2)), InvSize(2), tf("iv1"))
            iv2 = AbsorbVertex(CompVertex(identity, inpt(3,2)), InvSize(3), tf("iv2"))
            tv = StackingVertex(CompVertex(hcat, iv1, iv2), tf("tv"))
            io1 = AbsorbVertex(CompVertex(identity, tv), InvSize(5), tf("io1"))

            @test inputs(tv) == [iv1, iv2]
            @test outputs.(inputs(tv)) == [[tv], [tv]]
            @test outputs(iv1) == [tv]
            @test outputs(iv2) == [tv]

            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == [5]

            Δnout(iv2, -2)
            @test nin(io1) == [nout(tv)] == [sum(nin(tv))] == [nout(iv1) + nout(iv2)]  ==  [3]

            Δnin(io1, 3)
            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == [6]

            Δnout(iv1, 4)
            Δnin(io1, -8)
            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == [2]
            @test nout(iv1) == nout(iv2) == 1

            #Add another output
            io2 = AbsorbVertex(CompVertex(identity, tv), InvSize(2), tf("io2"))

            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == nin(io2) == [2]

            Δnout(iv1, 3)
            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == nin(io2) == [5]

            Δnin(io2, -2)
            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == nin(io2) == [3]

            Δnin(io1, 3)
            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == nin(io2) == [6]

            Δnin(tv, -1, missing)
            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == nin(io2) == [5]

            Δnin(tv, missing, 2)
            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == nin(io2) == [7]

            Δnout(tv, -1)
            @test [nout(iv1) + nout(iv2)] == [sum(nin(tv))] == [nout(tv)] == nin(io1) == nin(io2) == [6]

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
            @test [nout(iv)] == nin(inv) == [nout(inv)] == nin(io) == [5]

            Δnin(io, -2)
            @test [nout(iv)] == nin(inv) == [nout(inv)] == nin(io) == [3]

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

            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1)[] == 3

            Δnout(iv2, -2)
            @test nin(io1)[] == nout(inv) == nin(inv)[1] == nin(inv)[2]  == nout(iv1) == nout(iv2)  ==  1

            Δnin(io1, 3)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1)[] == 4

            #Add another output
            io2 = AbsorbVertex(CompVertex(identity, inv), InvSize(4))

            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1)[] == nin(io2)[] == 4

            Δnout(iv1, -3)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1)[] == nin(io2)[] == 1

            Δnin(io2, 2)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1)[] == nin(io2)[] == 3

            Δnout(iv2, 3)
            @test nout(iv1) == nout(iv2) == nin(inv)[1] == nin(inv)[2] == nout(inv) == nin(io1)[] == nin(io2)[] == 6

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
        rb(start, residual,name="add") = traitconf(tf(name)) >> start + residual
        iv(in; name="iv") = invariantvertex(identity, in, traitdecoration=tf(name))
        concd1(paths...;name="conc") = conc(paths..., dims=1, traitdecoration=tf(name))
        mm(nin, nout) = x -> x * reshape(collect(1:nin*nout), nin, nout)
        av(in, outsize, name="comp") = absorbvertex(mm(nout(in), outsize), outsize, in, traitdecoration = tf(name))
        function stack(start, nouts...; bname = "stack")
            # Can be done on one line with mapfoldl, but it is not pretty...
            next = start
            for i in 1:length(nouts)
                next = av(next, nouts[i], "$(bname)_$i")
            end
            return next
        end

        vnames(Δg::MetaDiGraph) = mapfoldl(e -> name(Δg[e.src, :vertex]) => name(Δg[e.dst, :vertex]) , vcat, edges(Δg))
        dirs(Δg::MetaDiGraph) =  mapfoldl(e -> get_prop(Δg, e, :direction), vcat, edges(Δg))

        @testset "Residual fork block" begin
            start = av(inputvertex("in", 3), 9, "start")
            p1 = stack(start, 3,4, bname = "p1")
            p2 = stack(start, 4,5, bname = "p2")
            resout = rb(start, concd1(p1, p2))
            out = av(resout, 9, "out")

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
            start = av(inputvertex("in", 3), 8, "start")
            split = av(start, 4, "split")
            p1 = iv(split, name="p1") #Just an identity vertex
            p2 = stack(split, 3,2,4, bname="p2")
            resout = rb(start, concd1(p1, p2))
            out = av(resout, 3, "out")

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
            start = av(inputvertex("in", 3), 4, "start")
            p1 = iv(start, name="p1")
            p2 = iv(start, name="p2")
            joined = concd1(p1, p2, name="join")
            out = av(joined, 3, "out")

            @test nout(joined) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the conc before resout. Simple dfs will
            # fail as one will hit the conc through p1 before having
            # resolved the path through p2.
            Δnout(start, -1)
            @test nin(out) == [2nout(start)] == [6]

            # Should basically undo the previous mutation
            @test minΔninfactor(out) == 2
            Δnin(out, +2)
            @test nin(out) == [2nout(start)] == [8]
        end

        @testset "Transparent residual fork block" begin
            start = av(inputvertex("in", 3), 8, "start")
            split = av(start, 4, "split")
            p1 = iv(split, name="p1")
            p2 = iv(split, name="p2")
            resout = rb(start, concd1(p1, p2, name="join"))
            out = av(resout, 3, "out")

            @test nout(resout) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the conc before resout. Simple dfs will
            # fail as one will hit the conc through p1 before having
            # resolved the path through p2.
            Δnout(split, -1)
            @test nin(out) == [nout(start)] == nin(split) == [6]

            # Should basically undo the previous mutation
            Δnin(out, +2)
            @test nin(out) == [nout(start)] == nin(split) == [8]

            @test minΔninfactor(out) == 2
            Δnout(start, -2)
            @test nin(out) == [nout(start)] == [6]
            @test nout(split) == 3
        end

        @testset "Transparent residual fork block with single absorbing path" begin
            start = av(inputvertex("in", 3), 8, "start")
            split = av(start, 3, "split")
            p1 = iv(split, name="p1")
            p2 = iv(split, name="p2")
            p3 = av(split, 2, "p3")
            resout = rb(start, concd1(p1, p2, p3, name="join"), "add")
            out = av(resout, 3, "out")

            @test nout(resout) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the conc before resout. Simple dfs will
            # fail as one will hit the conc through p1 before having
            # resolved the path through p2.
            Δnout(split, -1)
            @test nin(out) == [nout(start)] == nin(split) == [6]

            # Should basically undo the previous mutation
            Δnin(out, +2)
            @test nin(out) == [nout(start)] == nin(split) == [8]

            @test minΔninfactor(out) == 2
            Δnout(start, -2)
            @test nin(out) == [nout(start)] == [2*nout(split) + nout(p3)] == [6]
            @test nout(split) == 2
            @test nout(p3) == 2
        end

        @testset "StackingVertex maze" begin

            v1 = inpt(3, "in")
            v2 = av(v1, 3, "v2")
            v3 = av(v1, 5, "v3")
            v4 = av(v1, 7, "v4")
            v5 = av(v1, 11, "v5")
            v6 = av(v1, 13, "v6")

            sv1 = concd1(v2, v3, v4, name="sv1")
            sv2 = concd1(v5, v3, v4, name="sv2")
            sv3 = concd1(sv1, sv2, v6, name="sv3")

            o1 = av(sv1, 2, "o1")
            o2 = av(sv2, 2, "o2")
            o3 = av(sv3, 2, "o3")

            # Evil action! Must have understanding that change will propagate to sv1 from
            # both v2 and v3 and hold off updating sv1 when the first of them (v1 in this case) is updated. Must also hold off updating sv3 for the same reasons
            Δnout(sv2, -6)
            @test nin(sv1) == nout.(inputs(sv1)) == [3, 4, 5]
            @test nin(sv2) == nout.(inputs(sv2)) == [8, 4, 5]
            @test nin(sv3) == nout.(inputs(sv3)) == [12, 17, 13]

            # Tip: Turn use a SizeChangeLogger to verify correct output
            Δg = ΔnoutSizeGraph(sv2)
            @test [vnames(Δg) dirs(Δg)] == [
            "sv2"=>"v5"   Output();
            "sv2"=>"v3"   Output();
            "sv2"=>"sv3"  Input();
            "sv2"=>"v4"   Output();
            "sv2"=>"o2"   Input();
            "v3"=>"sv1"  Input();
            "sv1"=>"sv3"  Input();
            "sv1"=>"o1"   Input();
            "sv3"=>"o3"   Input();
            "v4"=>"sv1"  Input()]


            Δg = ΔninSizeGraph(sv2)
            @test [vnames(Δg) dirs(Δg)] == [
            "sv2"=>"v5"   Output();
            "sv2"=>"v3"   Output();
            "sv2"=>"sv3"  Input();
            "sv2"=>"v4"   Output();
            "sv2"=>"o2"   Input();
            "v3"=>"sv1"  Input();
            "sv1"=>"sv3"  Input();
            "sv1"=>"o1"   Input();
            "sv3"=>"o3"   Input();
            "v4"=>"sv1"  Input()]

            Δnin(sv2, 2, missing, 2)
            @test nin(sv1) == nout.(inputs(sv1)) == [3, 4, 7]
            @test nin(sv2) == nout.(inputs(sv2)) == [10, 4, 7]
            @test nin(sv3) == nout.(inputs(sv3)) == [14, 21, 13]

            Δg = ΔninSizeGraph(sv2, false, true ,false)
            @test [vnames(Δg) dirs(Δg)] == [
            "sv2"=>"v5"   Output();
            "sv2"=>"v4"   Output();
            "sv2"=>"sv3"  Input();
            "sv2"=>"o2"   Input();
             "v4"=>"sv1"  Input();
            "sv1"=>"sv3"  Input();
            "sv1"=>"o1"   Input();
            "sv3"=>"o3"   Input()]
        end

        @testset "SizeStack duplicate vertex cycle" begin
            v0 = inpt(3, "in")
            v1 = av(v0, 7, "v1")
            v2 = av(v0, 4, "v2")
            v3 = concd1(v1, v2, name="v3")
            v4 = concd1(v2, v3, name="v4")

            @test minΔnoutfactor(v4) == 2

            # Trouble as we might go Δnout(v4) -> Δnout(v2) -> Δnin(v3) and then exit at Δnout(v3) as v3 has already been vistited
            Δnout(v4, -4)

            @test nout(v4) == sum(nin(v4)) == nout(v2) + nout(v3) == 11
            @test nout(v3) == sum(nin(v3)) == nout(v1) + nout(v2) == 8

            Δnout(v2, 4)

            @test nout(v4) == sum(nin(v4)) == nout(v2) + nout(v3) == 19
            @test nout(v3) == sum(nin(v3)) == nout(v1) + nout(v2) == 12

            Δg = ΔnoutSizeGraph(v4)
            @test [vnames(Δg) dirs(Δg)] == [
            "v4" => "v2" Output();
            "v4" => "v3" Output();
            "v3" => "v1" Output()]

            Δg = ΔninSizeGraph(v3)
            @test [vnames(Δg) dirs(Δg)] == [
            "v3" => "v1" Output();
            "v3" => "v2" Output();
            "v3" => "v4" Input();
            "v2" => "v4" Input()]
        end

        @testset "Entangled SizeStack" begin
            v0 = inpt(2, "in")

            v1 = av(v0, 5, "v1")
            v2 = av(v0, 4, "v2")
            v3 = av(v0, 3, "v3")
            v4 = av(v0, 6, "v4")

            v5 = concd1(v1, v2, name="v5")
            v6 = concd1(v2, v3, name="v6")
            v7 = concd1(v3, v4, name="v7")

            v8 = concd1(v5, v6, name="v8")
            v9 = concd1(v6, v7, name="v9")

            v10 = rb(v8,v9, "v10")

            @test minΔnoutfactor(v10) == 2
            Δnout(v10, -2)

            @test nout(v10) == unique(nin(v10))[] == nout(v8) == nout(v9) == sum(nin(v8)) == sum(nin(v9)) == 14

            @test nin(v9) == nout.(inputs(v9)) == sum.(nin.(inputs(v9))) == [6, 8]
            @test nin(v8) == nout.(inputs(v8)) == sum.(nin.(inputs(v8))) == [8, 6]

            @test nin(v7) == nout.(inputs(v7)) == [2, 6]
            @test nin(v6) == nout.(inputs(v6)) == [4, 2]
            @test nin(v5) == nout.(inputs(v5)) == [4, 4]

        end
    end

    @testset "Size Mutation possibilities" begin
        # Helpers
        struct SizeConstraint constraint; end
        NaiveNASlib.minΔnoutfactor(c::SizeConstraint) = c.constraint
        NaiveNASlib.minΔninfactor(c::SizeConstraint) = c.constraint
        av(size, csize, in... ;name = "av") = absorbvertex(SizeConstraint(csize), size, in..., traitdecoration=tf(name))
        sv(in...; name="sv") = conc(in..., dims=1, traitdecoration = tf(name))
        iv(ins...; name="iv") = +(traitconf(tf(name)) >> ins[1], ins[2:end]...)

        @testset "InputSizeVertex" begin
            @test ismissing(minΔnoutfactor(inpt(3)))
            @test ismissing(minΔninfactor(inpt(3)))
        end

        @testset "SizeAbsorb" begin
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

        @testset "SizeStack multi inputs" begin
            v1 = av(6,3, inpt(3), name="v1")
            v2 = av(7,2, inpt(3), name="v2")
            v3 = av(8,2, inpt(3), name="v3")

            sv1 = sv(v1, v2, name="sv1")
            sv2 = sv(v3, v2, v1, v2, name="sv2")
            @test minΔnoutfactor(sv1) == 6
            @test minΔnoutfactor(sv2) == 12

            # Expect only v1 to change as size change is not compatible with v2
            Δnout(sv1, -3)
            @test nin(sv1) == [nout(v1), nout(v2)] == [3, 7]
            @test nin(sv2) == [nout(v3), nout(v2), nout(v1), nout(v2)] == [8, 7, 3, 7]

            # v1 can't change as it is too small already
            # v3 makes a larger change as it is larger than v1
            Δnout(sv2, -6)
            @test nin(sv1) == [nout(v1), nout(v2)] == [3, 7]
            @test nin(sv2) == [nout(v3), nout(v2), nout(v1), nout(v2)] == [2, 7, 3, 7]

            # Evil action! Must have understanding that change will propagate to sv1 from
            # both v1 and v2 and hold off updating sv1 when the first of them (v1 in this case) is updated
            Δnout(sv2, +9)
            @test nin(sv1) == [nout(v1), nout(v2)] == [6, 9]
            @test nin(sv2) == [nout(v3), nout(v2), nout(v1), nout(v2)] == [4, 9, 6, 9]
        end

        @testset "Stacked SizeStacks" begin
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

            v5 = av(10, 3, sv3, name="v5")
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
            Δnout(v2, -12)
            @test nout(v1) == 89
            @test nout(v2) == 80
            @test nout(v3) == 85
            @test nout(v4) == 85
            @test nout(sv1) == nout(v2) + nout(v3) == 165
            @test nout(sv2) == nout(sv1) + nout(v1) == 254
            @test [nout(sv3)] == [(nout(sv2) + nout(v4) + nout(v2))] == nin(v5) == [419]
        end

        @testset "SizeStack large Δ" begin
            v1 = av(100,2, inpt(3), name="v1")
            v2 = av(100,3, inpt(3), name="v1")
            vs = [av(100,1, inpt(3), name="v$i") for i in 1:10]

            sv1  = sv(v1, v2, vs..., name="sv1")

            # This would take forever to brute force...
            Δnout(sv1, -600)
            @test nout(sv1) == sum(nin(sv1)) == sum(nout.([v1, v2, vs...])) == 600
            @test nin(sv1) == nout.([v1, v2, vs...])
        end

        @testset "Stacked SizeInvariant" begin
            v1 = av(100,1, inpt(3), name="v1")
            v2 = av(100,2, inpt(3), name="v2")
            v3 = av(100,3, inpt(3), name="v3")
            v4 = av(100,5, inpt(3), name="v4")

            iv1 = iv(v2, v3, name="iv1")
            iv2 = iv(iv1, v1, name="iv2")
            iv3 = iv(iv2, v4, v2, name="iv3")

            # Everything thouches everything in this setup
            @test minΔnoutfactor(iv1) == minΔninfactor(iv1) == 1*2*3*5
            @test minΔnoutfactor(iv2) == minΔninfactor(iv2) == 1*2*3*5
            @test minΔnoutfactor(iv3) == minΔninfactor(iv3) == 1*2*3*5

            Δnout(iv3, -30)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 70
            @test nout(iv1) == nout(iv2) == nout(iv3) == 70

            v5 = av(10, 3, iv3, name="v5")

            Δnout(v1, 3)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 73
            @test [nout(iv1)] == [nout(iv2)] == [nout(iv3)] == nin(v5) == [73]

            # Evil action! Must have understanding that the change in v2 will propagate
            # to iv3 input 1 through iv1 and iv2 and hold off updating it through input 3
            Δnout(v2, -12)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 61
            @test [nout(iv1)] == [nout(iv2)] == [nout(iv3)] == nin(v5) == [61]
        end

        @testset "Stacked input to SizeInvariant" begin
            v0 = inpt(3, "in")
            v1 = av(20, 2, v0, name="v1")
            v2 = av(30, 3, v0, name="v2")
            v3 = sv(v1,v2,v1, name="v3")
            v4 = av(70, 5, v0, name="v4")
            v5 = iv(v4,v3, name="v5")
            v6 = av(10, 7, v5, name="v6")

            # Evilness: Invariant vertex must change all its inputs and therefore it must take their minΔnoutfactors into account when computing minΔninfactor.
            @test minΔnoutfactor(v4) == 2*2*3*5*7
        end

        @testset "Invariant input to SizeInvariant" begin
            v0 = inpt(3, "in")
            v1 = av(20, 2, v0, name="v1")
            v2 = av(20, 3, v0, name="v2")
            v3 = iv(v1,v2, name="v3")
            v4 = av(20, 5, v0, name="v4")
            v5 = iv(v4,v3, name="v5")
            v6 = av(10, 7, v5, name="v6")

            # Evilness: Infinite recursion without memoization
            @test minΔnoutfactor(v4) == 2*3*5*7
        end

        @testset "Immutable input to SizeInvariant after SizeStack" begin
            v0 = inpt(5, "in")
            v1 = av(2, 2, v0, name="v1")
            v2 = av(3, 3, v0, name="v2")
            v3 = sv(v1,v2, name="v3")
            v4 = iv(v0,v3, name="v4")
            v5 = av(10, 7, v4, name="v5")

            # Evilness: Δnout(v3) not possible as it'll "bounce" on v4 into v0 (which is immutable)
            @test ismissing(minΔnoutfactor(v3))
            @test_throws ErrorException Δnout(v3, 2)
        end

        @testset "SizeStack duplicate through SizeInvariant" begin
            v0 = inpt(5, "in")
            v1 = av(2, 2, v0, name="v1")
            v2 = iv(v1, name="v2")
            v3 = iv(v1, name="v3")
            v4 = sv(v2,v3, name="v4")
            v5 = iv(v4, name="v5")
            v6 = sv(v5, v2, name="v6")

            @test minΔnoutfactor(v6) == 3*2
            @test Δnout(v6, 12)

            @test nout(v6) == sum(nin(v6)) == 18
            @test nout(v5) == nin(v5)[] == 12
            @test nout(v4) == sum(nin(v4)) == 12
            @test nout(v1) == 6
        end

        @testset "SizeInvariant zig-zag" begin
                v0 = inpt(5, "in")
                v1 = av(2, 2, v0, name="v1")
                v2 = av(3, 3, v0, name="v2")
                v3 = sv(v1,v2, name="v3")
                function zigzag(vin1, sc, vin2=v0;name="zig")
                    vnew = av(nout(vin1), sc, vin2, name=name*"_new")
                    vout = iv(vnew, name=name*"_ivB")
                    vcon = iv(vout, vin1, name=name*"_ivA")
                    return vout
                end
                v4 = zigzag(v3, 5, name="z1")
                v5 = zigzag(v4, 7, name="z2")
                v6 = av(11, 11, v5, name="v6")

                expectedΔf = 2*3*5*7*11
                @test minΔnoutfactor(v1) == expectedΔf / 3 # 3 is size constraint for v2
                @test minΔnoutfactor(v2) == expectedΔf / 2 # 2 is size constraint for v1
                @test minΔnoutfactor(v3) == expectedΔf

                Δnout(v2, expectedΔf)
                @test nout(v3) == nout(v4) == nout(v5) == expectedΔf + 2+3
                @test nin(v4) == nout.(inputs(v4)) == [expectedΔf + 2+3]
                @test nin(v5) == nout.(inputs(v5)) == [expectedΔf + 2+3]
        end

        @testset "SizeStack duplicate SizeInvariant mini-zig-zag" begin
            v0 = inpt(5, "in")
            v1 = av(2, 2, v0, name="v1")
            v2 = av(2, 3, v0, name="v2")
            v3 = iv(v1, name="v3")
            v4 = iv(v3, v2, name="v4")
            v5 = av(3, 5, v3, name="v5")
            v6 = sv(v4,v4, name="v6")
            v7 = iv(v6, v6, name="v7")
            v8 = av(4, 7, v7, name="v8")

            expectedΔf = 2*2*3*5*7
            @test minΔnoutfactor(v6) == expectedΔf

            Δnout(v6, expectedΔf)
            @test nin(v8) == [nout(v7)] == [nout(v6)] == [4+expectedΔf]

            @test nin(v6) == [2 + expectedΔf ÷ 2, 2 + expectedΔf ÷ 2]
            @test nout(v1) == nout(v2) == 2 + expectedΔf ÷ 2
            @test nin(v5) == [2 + expectedΔf ÷ 2]
        end

        @testset "Deep SizeStack" begin
            v0 = inpt(3, "in")
            v1 = av(8, 1, v0, name="v1")
            v2 = av(4, 1, v0, name="v2")
            v3 = sv(v1,v2, name= "v3")
            pa1 = iv(v3, name="pa1")
            pb1 = iv(v3, name="pb1")
            pc1 = iv(v3, name="pc1")
            pd1 = av(5, 1, v3, name="pd1")
            pa1pa1 = iv(pa1, name="pa1pa1")
            pa1pb1 = iv(pa1, name="pa1pb1")
            pa2 = sv(pa1pa1, pa1pb1, name = "pa2")
            v4 = sv(pa2, pb1, pc1, pd1, name = "v4")

            @test minΔnoutfactor(v4) == 4
        end

        @testset "Fail invalid size change" begin
            v1 = av(100,3, inpt(3), name="v1")
            v2 = av(100,2, v1, name="v2")

            @test_throws ArgumentError Δnout(v1, 2)
            @test_throws ArgumentError Δnout(v1, 3)
            @test_throws ArgumentError Δnin(v2, 3)
            @test_throws ArgumentError Δnin(v2, 2)
        end

    end

    @testset "SizeChangeLogger" begin
        traitfun(name) = t -> SizeChangeLogger(NameInfoStr(), NamedTrait(t, name))

        av(in, size ;name = "av") = AbsorbVertex(CompVertex(identity, in), IoSize(nout(in), size), traitfun(name))
        sv(in...; name="sv") = StackingVertex(CompVertex(hcat, in...), traitfun(name))
        iv(in...; name="iv") = InvariantVertex(CompVertex(hcat, in...), traitfun(name))

        @testset "Log size change" begin
                v1 = inpt(3, "v1")
                v2 = av(v1, 10, name="v2")
                v3 = av(v2, 4, name="v3")

                @test_logs (:info, "Change nin of v3 by (3,)") (:info, "Change nout of v2 by 3") Δnin(v3, 3)

                @test_logs (:info, "Change nout of v2 by -4") (:info, "Change nin of v3 by (-4,)") Δnout(v2, -4)
        end
    end

    @testset "SizeChangeValidation" begin

        # Mock for creating invalid size changes
        struct Ignore <: MutationSizeTrait end
        NaiveNASlib.Δnin(::Ignore, v, Δ::Union{Missing, T}...; s) where T = Δnout.(inputs(v), Δ, s=s)
        NaiveNASlib.Δnout(::Ignore, v, Δ::T; s) where T = Δnin.(outputs(v), Δ, s=s)
        NaiveNASlib.Δnout_touches_nin(::Ignore, v, s) = NaiveNASlib.Δnout_touches_nin(SizeAbsorb(), v, s)
        NaiveNASlib.Δnout_touches_nin(::Ignore, v, from, s) = NaiveNASlib.Δnout_touches_nin(SizeAbsorb(), v, from, s)
        NaiveNASlib.Δnin_touches_nin(::Ignore, v, s) = NaiveNASlib.Δnin_touches_nin(SizeAbsorb(), v, s)
        NaiveNASlib.Δnin_touches_nin(::Ignore, v, from, s) = NaiveNASlib.Δnin_touches_nin(SizeAbsorb(), v, from, s)
        NaiveNASlib.update_state_nin!(t::Ignore, s, v, from) = NaiveNASlib.update_state_nin!(SizeAbsorb(), s, v, from)

        v1 = inpt(3, "v1")
        v2 = MutationVertex(CompVertex(identity, v1), IoSize(3, 5), Ignore()) # Note, does not catch size errors
        v3 = AbsorbVertex(CompVertex(identity, v2), IoSize(5,7), tf("v3"))
        v4 = MutationVertex(CompVertex(identity, v3), IoSize(3, 5), Ignore())

        @test_throws ArgumentError Δnout(v2, 2)
        @test_throws ArgumentError Δnin(v3, -3)
        @test_throws ArgumentError Δnin(v4, 5)
        @test_throws ArgumentError Δnout(v3, -7)

        v5 = AbsorbVertex(CompVertex(identity, v3), IoSize(nout(v3),3), tf("v5"))
        Δnin(v5, -1)
        @test nin(v5) == [nout(v3)] == [4]

        # Too many Δs!
        @test_throws ArgumentError Δnin(v5, 1, 1,s=NaiveNASlib.VisitState{Int}(v5, 1))
    end

    @testset "Mutate tricky structures JuMP" begin

        set_defaultΔNoutStrategy(DefaultJuMPΔSizeStrategy())
        set_defaultΔNinStrategy(DefaultJuMPΔSizeStrategy())

        ## Helper functions
        rb(start, residual,name="add") = traitconf(tf(name)) >> start + residual
        iv(in; name="iv") = invariantvertex(identity, in, traitdecoration=tf(name))
        concd1(paths...;name="conc") = conc(paths..., dims=1, traitdecoration=tf(name))
        mm(nin, nout) = x -> x * reshape(collect(1:nin*nout), nin, nout)
        av(in, outsize, name="comp") = absorbvertex(mm(nout(in), outsize), outsize, in, traitdecoration = tf(name))
        function stack(start, nouts...; bname = "stack")
            # Can be done on one line with mapfoldl, but it is not pretty...
            next = start
            for i in 1:length(nouts)
                next = av(next, nouts[i], "$(bname)_$i")
            end
            return next
        end

        @testset "Residual fork block" begin
            start = av(inputvertex("in", 3), 9, "start")
            p1 = stack(start, 3,4, bname = "p1")
            p2 = stack(start, 4,5, bname = "p2")
            resout = rb(start, concd1(p1, p2))
            out = av(resout, 9, "out")

            @test nout(resout) == 9

            # Propagates to out, start outputs of start and also to p1 as objective is minimized by increasing p1 by 1
            Δnout(p2, -2)
            @test nout(p2) == 3
            @test nin(out) == [nout(start)] == [8]
            @test nout(p2) + nout(p1) == 8
            #outputs(start) = first vertex in p1, p2 and resout (which has two inputs)
            @test foldl(vcat, nin.(outputs(start))) == [8, 8, 8, 8]

            Δnin(out, +2)
            @test nin(out) == [nout(start)] == [10]
            @test foldl(vcat, nin.(outputs(start))) == [10, 10, 10, 10]
        end

        @testset "Half transparent residual fork block" begin
            start = av(inputvertex("in", 3), 8, "start")
            split = av(start, 4, "split")
            p1 = iv(split, name="p1") #Just an identity vertex
            p2 = stack(split, 3,2,4, bname="p2")
            resout = rb(start, concd1(p1, p2))
            out = av(resout, 3, "out")

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
            start = av(inputvertex("in", 3), 4, "start")
            p1 = iv(start, name="p1")
            p2 = iv(start, name="p2")
            joined = concd1(p1, p2, name="join")
            out = av(joined, 3, "out")

            @test nout(joined) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the conc before resout. Simple dfs will
            # fail as one will hit the conc through p1 before having
            # resolved the path through p2.
            Δnout(start, -1)
            @test nin(out) == [2nout(start)] == [6]

            # Should basically undo the previous mutation
            @test minΔninfactor(out) == 2
            Δnin(out, +2)
            @test nin(out) == [2nout(start)] == [8]
        end

        @testset "Transparent residual fork block" begin
            start = av(inputvertex("in", 3), 8, "start")
            split = av(start, 4, "split")
            p1 = iv(split, name="p1")
            p2 = iv(split, name="p2")
            resout = rb(start, concd1(p1, p2, name="join"))
            out = av(resout, 3, "out")

            @test nout(resout) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the conc before resout. Simple dfs will
            # fail as one will hit the conc through p1 before having
            # resolved the path through p2.
            Δnout(split, -1)
            @test nin(out) == [nout(start)] == nin(split) == [6]

            # Should basically undo the previous mutation
            Δnin(out, +2)
            @test nin(out) == [nout(start)] == nin(split) == [8]

            @test minΔninfactor(out) == 2
            Δnout(start, -2)
            @test nin(out) == [nout(start)] == [6]
            @test nout(split) == 3
        end

        @testset "Transparent residual fork block with single absorbing path" begin
            start = av(inputvertex("in", 3), 8, "start")
            split = av(start, 3, "split")
            p1 = iv(split, name="p1")
            p2 = iv(split, name="p2")
            p3 = av(split, 2, "p3")
            resout = rb(start, concd1(p1, p2, p3, name="join"), "add")
            out = av(resout, 3, "out")

            @test nout(resout) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the conc before resout. Simple dfs will
            # fail as one will hit the conc through p1 before having
            # resolved the path through p2.
            Δnout(split, -1)
            @test nin(out) == [nout(start)] == nin(split) == [6]

            # Should basically undo the previous mutation
            Δnin(out, +2)
            @test nin(out) == [nout(start)] == nin(split) == [8]

            @test minΔninfactor(out) == 2
            Δnout(start, -2)
            @test nin(out) == [nout(start)] == [2*nout(split) + nout(p3)] == [6]
            @test nout(split) == 2
            @test nout(p3) == 2
        end
        set_defaultΔNoutStrategy(ΔNoutLegacy())
        set_defaultΔNinStrategy(ΔNinLegacy())
    end


    @testset "Size Mutation possibilities using JuMP" begin
        set_defaultΔNoutStrategy(DefaultJuMPΔSizeStrategy())
        set_defaultΔNinStrategy(DefaultJuMPΔSizeStrategy())
        # Helpers
        struct SizeConstraint constraint; end
        NaiveNASlib.minΔnoutfactor(c::SizeConstraint) = c.constraint
        NaiveNASlib.minΔninfactor(c::SizeConstraint) = c.constraint
        function NaiveNASlib.compconstraint!(s, c::SizeConstraint, data)
            fv_out = JuMP.@variable(data.model, integer=true)
            JuMP.@constraint(data.model, c.constraint * fv_out ==  nout(data.vertex) - data.noutdict[data.vertex])

            ins = inputs(data.vertex)
            fv_in = JuMP.@variable(data.model, [1:length(ins)], integer=true)
            JuMP.@constraint(data.model, [i=1:length(ins)], c.constraint * fv_in[i] ==  nout(ins[i]) - data.noutdict[ins[i]])
        end
        av(size, csize, in... ;name = "av") = absorbvertex(SizeConstraint(csize), size, in..., traitdecoration=tf(name))
        sv(in...; name="sv") = conc(in..., dims=1, traitdecoration = tf(name))
        iv(ins...; name="iv") = +(traitconf(tf(name)) >> ins[1], ins[2:end]...)

        @testset "SizeStack multi inputs" begin
            v1 = av(6,3, inpt(3), name="v1")
            v2 = av(7,2, inpt(3), name="v2")
            v3 = av(8,2, inpt(3), name="v3")

            sv1 = sv(v1, v2, name="sv1")
            sv2 = sv(v3, v2, v1, v2, name="sv2")
            @test minΔnoutfactor(sv1) == 6
            @test minΔnoutfactor(sv2) == 12

            # Expect only v1 to change as size change is not compatible with v2
            Δnout(sv1, -3)
            @test nin(sv1) == [nout(v1), nout(v2)] == [3, 7]
            @test nin(sv2) == [nout(v3), nout(v2), nout(v1), nout(v2)] == [8, 7, 3, 7]
            @test nout(sv1) == sum(nin(sv1))
            @test nout(sv2) == sum(nin(sv2))

            # v1 can't change as it is too small already
            Δnout(sv2, -6)
            @test nin(sv1) == [nout(v1), nout(v2)] == [3, 5]
            @test nin(sv2) == [nout(v3), nout(v2), nout(v1), nout(v2)] == [6, 5, 3, 5]
            @test nout(sv1) == sum(nin(sv1))
            @test nout(sv2) == sum(nin(sv2))


            # Evil action! Must have understanding that change will propagate to sv1 from
            # both v1 and v2
            Δnout(sv2, +9)
            @test nin(sv1) == [nout(v1), nout(v2)] == [6, 7]
            @test nin(sv2) == [nout(v3), nout(v2), nout(v1), nout(v2)] == [8, 7, 6, 7]
            @test nout(sv1) == sum(nin(sv1))
            @test nout(sv2) == sum(nin(sv2))

        end

        @testset "Stacked SizeStacks" begin
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
            @test nin(sv1) == nout.(inputs(sv1)) == [84, 97]
            @test nin(sv2) == nout.(inputs(sv2)) == [181, 90]
            @test nin(sv3) == nout.(inputs(sv3)) == [271, 85, 84]
            @test nout(sv1) == nout(v2) + nout(v3) == 181
            @test nout(sv2) == nout(sv1) + nout(v1) == 271
            @test nout(sv3) == nout(sv2) + nout(v4) + nout(v2) == 440

            v5 = av(10, 3, sv3, name="v5")
            @test minΔnoutfactor(sv1) == minΔninfactor(sv1) == 6
            @test minΔnoutfactor(sv2) == minΔninfactor(sv2)== 6
            @test minΔnoutfactor(sv3) == minΔninfactor(sv3) == 60

            Δnout(v1, 3)
            @test nin(sv1) == nout.(inputs(sv1)) == [84, 97]
            @test nin(sv2) == nout.(inputs(sv2)) == [181, 93]
            @test nin(sv3) == nout.(inputs(sv3)) == [274, 85, 84]
            @test nout(sv1) == nout(v2) + nout(v3) == 181
            @test nout(sv2) == nout(sv1) + nout(v1) == 274
            @test [nout(sv3)] == [(nout(sv2) + nout(v4) + nout(v2))] == nin(v5) == [443]

            # Evil action! Must have understanding that the change in v2 will propagate
            # to sv3 input 1 through sv1 and sv2
            Δnout(v2, -12)
            @test nin(sv1) == nout.(inputs(sv1)) == [72, 100]
            @test nin(sv2) == nout.(inputs(sv2)) == [172, 96]
            @test nin(sv3) == nout.(inputs(sv3)) == [268, 85, 72]
            @test nout(sv1) == nout(v2) + nout(v3) == 172
            @test nout(sv2) == nout(sv1) + nout(v1) == 268
            @test [nout(sv3)] == [(nout(sv2) + nout(v4) + nout(v2))] == nin(v5) == [425]
        end

        @testset "SizeStack large Δ" begin
            v1 = av(100,2, inpt(3), name="v1")
            v2 = av(100,3, inpt(3), name="v1")
            vs = [av(100,1, inpt(3), name="v$i") for i in 1:10]

            sv1  = sv(v1, v2, vs..., name="sv1")

            # This would take forever to brute force...
            Δnout(sv1, -600)
            @test nout(sv1) == sum(nin(sv1)) == sum(nout.([v1, v2, vs...])) == 600
            @test nin(sv1) == nout.([v1, v2, vs...])
        end

        @testset "SizeInvariant multi input" begin
            v1 = av(10,1, inpt(3), name="v1")

            iv1 = iv(v1, name="iv1")
            iv2 = iv(iv1, v1, name="iv2")

            Δnout(iv2, -2)

            @test nout(iv2) == nout(iv1) == nout(v1) == 8
            @test nin(iv2) == [nout(iv1), nout(v1)] == [8, 8]
            @test nin(iv1) == [nout(iv1)] == [8]
        end

        @testset "SizeInvariant multi SizeStack input" begin
            v1 = av(7,1, inpt(3), name="v1")
            v2 = av(13,1, inpt(3), name="v2")

            sv1 = sv(v1, v2, name="sv1")
            sv2 = sv(v1, v2, name="sv2")

            iv1 = iv(sv1, sv2, name="iv1")
            iv2 = iv(sv1, sv2, name="iv2")
            iv3 = iv(iv1, iv2, name="iv3")

            Δnout(iv3, -4)

            @test nout(iv3) == nout(iv2) == nout(iv1) == nout(sv2) == nout(sv1) == nout(v2) + nout(v1) == 16
            @test nin(iv3) == nin(iv2) == nin(iv3) == [nout(sv1), nout(sv2)] == [16, 16]
            @test nin(sv2) == nin(sv1) == [nout(v1), nout(v2)] == [6, 10]
        end

        @testset "Stacked SizeInvariant" begin
            v1 = av(100,1, inpt(3), name="v1")
            v2 = av(100,2, inpt(3), name="v2")
            v3 = av(100,3, inpt(3), name="v3")
            v4 = av(100,5, inpt(3), name="v4")

            iv1 = iv(v2, v3, name="iv1")
            iv2 = iv(iv1, v1, name="iv2")
            iv3 = iv(iv2, v4, v2, name="iv3")

            # Everything thouches everything in this setup
            @test minΔnoutfactor(iv1) == minΔninfactor(iv1) == 1*2*3*5
            @test minΔnoutfactor(iv2) == minΔninfactor(iv2) == 1*2*3*5
            @test minΔnoutfactor(iv3) == minΔninfactor(iv3) == 1*2*3*5

            Δnout(iv3, -30)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 70
            @test nout(iv1) == nout(iv2) == nout(iv3) == 70

            v5 = av(10, 3, iv3, name="v5")

            Δnout(v1, 60)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 130
            @test [nout(iv1)] == [nout(iv2)] == [nout(iv3)] == nin(v5) == [130]

            # Evil action! Must have understanding that the change in v2 will propagate
            # to iv3 input 1 through iv1 and iv2 and hold off updating it through input 3
            Δnout(v2, -30)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 100
            @test [nout(iv1)] == [nout(iv2)] == [nout(iv3)] == nin(v5) == [100]
        end

        @testset "Stacked input to SizeInvariant" begin
            v0 = inpt(3, "in")
            v1 = av(20, 2, v0, name="v1")
            v2 = av(30, 3, v0, name="v2")
            v3 = sv(v1,v2,v1, name="v3")
            v4 = av(70, 5, v0, name="v4")
            v5 = iv(v4,v3, name="v5")
            v6 = av(10, 7, v5, name="v6")

            # Evilness: Invariant vertex must change all its inputs and therefore it must take their minΔnoutfactors into account when computing minΔninfactor.
            @test minΔnoutfactor(v4) == 2*2*3*5*7

            Δnout(v5, 2*2*3*5*7)

            @test nout(v5) == nout(v4) == nout(v3) == 490
            @test nin(v5) == [nout(v4), nout(v3)] == [490, 490]
            @test nin(v3) == [nout(v1), nout(v2), nout(v1)] == [152, 186, 152]
        end

        @testset "Invariant input to SizeInvariant" begin
            v0 = inpt(3, "in")
            v1 = av(20, 2, v0, name="v1")
            v2 = av(20, 3, v0, name="v2")
            v3 = iv(v1,v2, name="v3")
            v4 = av(20, 5, v0, name="v4")
            v5 = iv(v4,v3, name="v5")
            v6 = av(10, 7, v5, name="v6")

            # Evilness: Infinite recursion without memoization
            @test minΔnoutfactor(v4) == 2*3*5*7

            Δnout(v5, 2*3*5*7)

            @test nout(v5) == nout(v4) == nout(v3) == 230
            @test nin(v5) == [nout(v4), nout(v3)] == nin(v3) == [nout(v1), nout(v2)] == [230, 230]
        end

        @testset "SizeStack duplicate through SizeInvariant" begin
            v0 = inpt(5, "in")
            v1 = av(2, 2, v0, name="v1")
            v2 = iv(v1, name="v2")
            v3 = iv(v1, name="v3")
            v4 = sv(v2,v3, name="v4")
            v5 = iv(v4, name="v5")
            v6 = sv(v5, v2, name="v6")

            @test minΔnoutfactor(v6) == 3*2
            Δnout(v6, 12)

            @test nout(v6) == sum(nin(v6)) == 18
            @test nout(v5) == nin(v5)[] == 12
            @test nout(v4) == sum(nin(v4)) == 12
            @test nout(v1) == 6
        end

        @testset "SizeInvariant zig-zag" begin
                v0 = inpt(5, "in")
                v1 = av(2, 2, v0, name="v1")
                v2 = av(3, 3, v0, name="v2")
                v3 = sv(v1,v2, name="v3")
                function zigzag(vin1, sc, vin2=v0;name="zig")
                    vnew = av(nout(vin1), sc, vin2, name=name*"_new")
                    vout = iv(vnew, name=name*"_ivB")
                    vcon = iv(vout, vin1, name=name*"_ivA")
                    return vout
                end
                v4 = zigzag(v3, 5, name="z1")
                v5 = zigzag(v4, 7, name="z2")
                v6 = av(11, 11, v5, name="v6")

                expectedΔf = 2*3*5*7*11
                @test minΔnoutfactor(v1) == expectedΔf / 3 # 3 is size constraint for v2
                @test minΔnoutfactor(v2) == expectedΔf / 2 # 2 is size constraint for v1
                @test minΔnoutfactor(v3) == expectedΔf

                Δnout(v2, expectedΔf)
                @test nout(v3) == nout(v4) == nout(v5) == expectedΔf + 2+3
                @test nin(v4) == nout.(inputs(v4)) == [expectedΔf + 2+3]
                @test nin(v5) == nout.(inputs(v5)) == [expectedΔf + 2+3]
        end

        @testset "SizeStack duplicate SizeInvariant mini-zig-zag" begin
            v0 = inpt(5, "in")
            v1 = av(2, 2, v0, name="v1")
            v2 = av(2, 3, v0, name="v2")
            v3 = iv(v1, name="v3")
            v4 = iv(v3, v2, name="v4")
            v5 = av(3, 5, v3, name="v5")
            v6 = sv(v4,v4, name="v6")
            v7 = iv(v6, v6, name="v7")
            v8 = av(4, 7, v7, name="v8")

            expectedΔf = 2*2*3*5*7
            @test minΔnoutfactor(v6) == expectedΔf

            Δnout(v6, expectedΔf)
            @test nin(v8) == [nout(v7)] == [nout(v6)] == [4+expectedΔf]

            @test nin(v6) == [2 + expectedΔf ÷ 2, 2 + expectedΔf ÷ 2]
            @test nout(v1) == nout(v2) == 2 + expectedΔf ÷ 2
            @test nin(v5) == [2 + expectedΔf ÷ 2]
        end

        @testset "Deep SizeStack" begin
            v0 = inpt(3, "in")
            v1 = av(8, 1, v0, name="v1")
            v2 = av(4, 1, v0, name="v2")
            v3 = sv(v1,v2, name= "v3")
            pa1 = iv(v3, name="pa1")
            pb1 = iv(v3, name="pb1")
            pc1 = iv(v3, name="pc1")
            pd1 = av(5, 1, v3, name="pd1")
            pa1pa1 = iv(pa1, name="pa1pa1")
            pa1pb1 = iv(pa1, name="pa1pb1")
            pa2 = sv(pa1pa1, pa1pb1, name = "pa2")
            v4 = sv(pa2, pb1, pc1, pd1, name = "v4")

            Δnout(v4, 8)

            @test minΔnoutfactor(v4) == 4
        end

        @testset "Fail invalid size change" begin
            v1 = av(100,3, inpt(3), name="v1")
            v2 = av(100,2, v1, name="v2")

            # TODO: How to also @test_logs (:warn, "MIP couldn't be solved to optimality. Terminated with status: INFEASIBLE") ?

            @test_logs (:warn, "MIP couldn't be solved to optimality. Terminated with status: INFEASIBLE")  (@test_throws ErrorException Δnout(v1, 2))
            @test_logs (:warn, "MIP couldn't be solved to optimality. Terminated with status: INFEASIBLE")  (@test_throws ErrorException Δnout(v1, 3))

            @test_logs (:warn, "MIP couldn't be solved to optimality. Terminated with status: INFEASIBLE")  (@test_throws ErrorException Δnin(v2, 3))
            @test_logs (:warn, "MIP couldn't be solved to optimality. Terminated with status: INFEASIBLE")  (@test_throws ErrorException Δnin(v2, 2))
        end

        set_defaultΔNoutStrategy(ΔNoutLegacy())
        set_defaultΔNinStrategy(ΔNinLegacy())

    end

    @testset "SizeChangeLogger" begin
        traitfun(name) = t -> SizeChangeLogger(NameInfoStr(), NamedTrait(t, name))

        av(in, size ;name = "av") = AbsorbVertex(CompVertex(identity, in), IoSize(nout(in), size), traitfun(name))
        sv(in...; name="sv") = StackingVertex(CompVertex(hcat, in...), traitfun(name))
        iv(in...; name="iv") = InvariantVertex(CompVertex(hcat, in...), traitfun(name))

        @testset "Log size change" begin
                v1 = inpt(3, "v1")
                v2 = av(v1, 10, name="v2")
                v3 = av(v2, 4, name="v3")

                @test_logs (:info, "Change nin of v3 by (3,)") (:info, "Change nout of v2 by 3") Δnin(v3, 3)

                @test_logs (:info, "Change nout of v2 by -4") (:info, "Change nin of v3 by (-4,)") Δnout(v2, -4)
        end
    end

end
