import JuMP

@testset "Size mutations" begin

    inpt(size, id="in") = inputvertex(id, size)
    nt(name) = t -> NamedTrait(t, name)
    tf(name) = t -> nt(name)(SizeChangeValidation(t))
    cc(in...; name="cc") = conc(in..., dims=1, traitdecoration = tf(name))
    ea(ins...; name="ea") = +(traitconf(tf(name)) >> ins[1], ins[2:end]...)

    @testset "Basic tests" begin

        av(in, outsize, name="av") = absorbvertex(SizeDummy(nout(in), outsize), in; traitdecoration=tf(name))

        @testset "AbsorbVertex" begin
            v0 = av(inpt(2), 2, "iv")
            v1 = av(v0, 3, "v1")

            @test nin(v1) == [2]
            @test nout(v1) == 3

            @test outputs.(inputs(v1)) == [[v1]]

            Δnin(v1, 2)
            @test nin(v1) == [4]

            Δnin(v1, -3)
            @test nin(v1) == [1]

            Δnout!(v1, -2)
            @test nout(v1) == 1

            # Add one vertex and see that change propagates
            v2 = av(v1, 4, "v2")
            @test nout(v1) == nin(v2)[1]
            @test outputs(v1) == [v2]
            @test inputs(v2) == [v1]

            Δnin(v2, 4)
            @test nout(v1) == nin(v2)[1] == 5

            Δnout!(v1, -2)
            @test nout(v1) == nin(v2)[1] == 3

            # Fork of v1 into a new vertex
            v3 = av(v1, 2, "v3")
            @test outputs(v1) == [v2, v3]
            @test inputs(v3) == inputs(v2) == [v1]

            Δnout!(v1, -2)
            @test nout(v1) == nin(v2)[1] == nin(v3)[1] == 1

            Δnin(v3, 3)
            @test nout(v1) == nin(v2)[1] == nin(v3)[1] == 4

            Δnin(v2, -2)
            @test nout(v1) == nin(v2)[1] == nin(v3)[1] == 2

            v0c = clone(v0, inputs(v0)[1])
            v1c = clone(v1, v0c)
            v2c = clone(v2, v1c)
            v3c = clone(v3, v1c)

            @test issame(v0, v0c)
            @test issame(v1, v1c)
            @test issame(v2, v2c)
            @test issame(v3, v3c)
        end

        @testset "StackingVertex" begin

            @testset "StackingVertex 1 to 1" begin

                v0 = av(inpt(2), 2, "v0")
                v1 = cc(v0)
                v2 = av(v1, 2,"v2")

                @test outputs.(inputs(v2)) == [[v2]]
                @test outputs.(inputs(v1)) == [[v1]]
                @test outputs(v0) == [v1]
                @test neighbours(Both(), v1) == [v0, v2]

                Δnout!(v0, 3)
                @test [nout(v0)] == nin(v1) == [nout(v1)] == nin(v2) == [5]

                Δnin(v2, -2)
                @test [nout(v0)] == nin(v1) == [nout(v1)] == nin(v2) == [3]

                Δnin(v1, +1)
                @test [nout(v0)] == nin(v1) == [nout(v1)] == nin(v2) == [4]

                Δnout!(v1, -1)
                @test [nout(v0)] == nin(v1) == [nout(v1)] == nin(v2) == [3]

                v0c = clone(v0, inputs(v0)[])
                v1c = clone(v1, v0c)
                v2c = clone(v2, v1c)

                @test issame(v0, v0c)
                @test issame(v1, v1c)
                @test issame(v2, v2c)
            end

            @testset "StackingVertex 2 inputs" begin
                # Try with two inputs to StackingVertex
                v1 = av(inpt(2,"in1"), 2, "v1")
                v2 = av(inpt(3,"in2"), 3, "v2")
                v3 = cc(v1, v2)
                v4 = av(v3, 5, "v4")

                @test inputs(v3) == [v1, v2]
                @test outputs.(inputs(v3)) == [[v3], [v3]]
                @test outputs(v1) == [v3]
                @test outputs(v2) == [v3]
                @test neighbours(Both(), v3) == [v1, v2, v4]

                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == [5]

                Δnout!(v2, -2)
                @test nin(v4) == [nout(v3)] == [sum(nin(v3))] == [nout(v1) + nout(v2)]  ==  [3]

                Δnin(v4, 3)
                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == [6]

                Δnout!(v1, 4)
                Δnin(v4, -8)
                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == [2]
                @test nout(v1) == nout(v2) == 1

                #Add another output
                v5 = av(v3, 2, "v5")

                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == nin(v5) == [2]

                Δnout!(v1, 3)
                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == nin(v5) == [5]

                Δnin(v5, -2)
                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == nin(v5) == [3]

                Δnin(v4, 3)
                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == nin(v5) == [6]

                Δnin(v3, -1, missing)
                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == nin(v5) == [5]

                Δnin(v3, missing, 2)
                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == nin(v5) == [7]

                Δnout!(v3, -1)
                @test [nout(v1) + nout(v2)] == [sum(nin(v3))] == [nout(v3)] == nin(v4) == nin(v5) == [6]

                v1c = clone(v1, inputs(v1)[])
                v2c = clone(v2, inputs(v2)[])
                v3c = clone(v3, v1c, v2c)
                v4c = clone(v4, v3c)
                v5c = clone(v5, v3c)

                @test issame(v1, v1c)
                @test issame(v2, v2c)
                @test issame(v3, v3c)
                @test issame(v4, v4c)
                @test issame(v5, v5c)
            end

        end
        @testset "InvariantVertex" begin

            @testset "InvariantVertex 1 to 1" begin
                #Behaviour is identical to StackingVertex in this case
                v1 = av(inpt(2), 2, "v1")
                v2 = ea(v1)
                v3 = av(v2, 2, "v3")

                @test outputs.(inputs(v3)) == [[v3]]
                @test outputs.(inputs(v2)) == [[v2]]
                @test outputs(v1) == [v2]

                Δnout!(v1, 3)
                @test [nout(v1)] == nin(v2) == [nout(v2)] == nin(v3) == [5]

                Δnin(v3, -2)
                @test [nout(v1)] == nin(v2) == [nout(v2)] == nin(v3) == [3]

                v1c = clone(v1, inputs(v1)[])
                v2c = clone(v2, v1c)
                v3c = clone(v3, v2c)

                @test issame(v1, v1c)
                @test issame(v2, v2c)
                @test issame(v3, v3c)
            end

            @testset "invariantvertex 2 inputs" begin
                # Try with two inputs a v3ariantVertex
                v1 = av(inpt(2, "in1"), 3, "v1")
                v2 = av(inpt(2, "in2"), 3, "v2")
                v3 = ea(v1, v2)
                v4 = av(v3, 5, "v4")

                @test inputs(v3) == [v1, v2]
                @test outputs.(inputs(v3)) == [[v3], [v3]]
                @test outputs(v1) == [v3]
                @test outputs(v2) == [v3]

                @test nout(v1) == nout(v2) == nin(v3)[1] == nin(v3)[2] == nout(v3) == nin(v4)[] == 3

                Δnout!(v2, -2)
                @test nin(v4)[] == nout(v3) == nin(v3)[1] == nin(v3)[2]  == nout(v1) == nout(v2)  ==  1

                Δnin(v4, 3)
                @test nout(v1) == nout(v2) == nin(v3)[1] == nin(v3)[2] == nout(v3) == nin(v4)[] == 4

                #Add another output
                v5 = av(v3, 4, "v5")

                @test nout(v1) == nout(v2) == nin(v3)[1] == nin(v3)[2] == nout(v3) == nin(v4)[] == nin(v5)[] == 4

                Δnout!(v1, -3)
                @test nout(v1) == nout(v2) == nin(v3)[1] == nin(v3)[2] == nout(v3) == nin(v4)[] == nin(v5)[] == 1

                Δnin(v5, 2)
                @test nout(v1) == nout(v2) == nin(v3)[1] == nin(v3)[2] == nout(v3) == nin(v4)[] == nin(v5)[] == 3

                Δnout!(v2, 3)
                @test nout(v1) == nout(v2) == nin(v3)[1] == nin(v3)[2] == nout(v3) == nin(v4)[] == nin(v5)[] == 6

                v1c = clone(v1, inputs(v1)[])
                v2c = clone(v2, inputs(v2)[])
                v3c = clone(v3, v1c, v2c)
                v4c = clone(v4, v3c)
                v5c = clone(v5, v3c)

                @test issame(v1, v1c)
                @test issame(v2, v2c)
                @test issame(v3, v3c)
                @test issame(v4, v4c)
                @test issame(v5, v5c)
            end
        end

        @testset "SizeChangeValidation" begin
            import NaiveNASlib: ΔSizeFailError

            # Mock for creating invalid size changes
            struct Ignore{T}
                f::T
            end
            NaiveNASlib.nout(i::Ignore) = nout(i.f)
            NaiveNASlib.nin(i::Ignore) = nin(i.f)
            NaiveNASlib.Δsize!(::Ignore, ::AbstractVector{<:Integer}, ::Integer) = nothing
            NaiveNASlib.Δsizetype(i::Ignore) = NaiveNASlib.Δsizetype(i.f)

            v1 = inpt(3, "v1")
            v2 = absorbvertex(Ignore(SizeDummy(nout(v1), 5)), v1; traitdecoration=tf("v2"))
            v3 = av(v2, 7, "v3")
            v4 = absorbvertex(Ignore(SizeDummy(nout(v3), 5)), v3; traitdecoration=tf("v4"))

            @test_throws ΔSizeFailError Δnout!(v2, 2)
            @test_throws ΔSizeFailError Δnin(v3, -3)
            @test_throws ΔSizeFailError Δnin(v4, 5)
            @test_throws ΔSizeFailError Δnout!(v3, -7)

            # Too many Δs!
            @test_throws AssertionError Δnin(v3, 1, 1)
        end

        @testset "Fail mutate immutable" begin
            import NaiveNASlib: ΔSizeFailError

            v0 = inpt(3)
            v1 = av(v0, 5, "v1")

            @test_throws ΔSizeFailError Δsize!(ΔNoutExact(v0, 2;fallback=ThrowΔSizeFailError("Success?!")), [v0, v1])
            @test [nout(v0)] == nin(v1) == [3]

            @test_throws ΔSizeFailError Δsize!(ΔNinExact(v1, -2; fallback=ThrowΔSizeFailError("Success?!")), [v1, v0])
            @test [nout(v0)] == nin(v1) == [3]
        end
    end

    @testset "Mutate tricky structures" begin

        ## Helper functions
        iv(in; name="iv") = invariantvertex(identity, in; traitdecoration=tf(name))
        concd1(paths...;name="conc") = conc(paths..., dims=1, traitdecoration=tf(name))
        av(in, outsize, name="comp") = absorbvertex(SizeDummy(nout(in), outsize), in, traitdecoration = tf(name))
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
            resout = "add" >> start + concd1(p1, p2)
            out = av(resout, 9, "out")

            @test nout(resout) == 9

            # Propagates to out, start outputs of start and also to p1 as objective is minimized by increasing p1 by 1
            @test Δnout!(p2, -2)
            @test nout(p2) == 3
            @test nin(out) == [nout(start)] == [8]
            @test nout(p2) + nout(p1) == 8
            #outputs(start) = first vertex in p1, p2 and resout (which has two inputs)
            @test reduce(vcat, nin.(outputs(start))) == [8, 8, 8, 8]

            @test Δnin(out, 2)
            @test nin(out) == [nout(start)] == [10]
            @test reduce(vcat, nin.(outputs(start))) == [10, 10, 10, 10]
        end

        @testset "Half transparent residual fork block" begin
            start = av(inputvertex("in", 3), 8, "start")
            split = av(start, 4, "split")
            p1 = iv(split, name="p1") #Just an identity vertex
            p2 = stack(split, 3,2,4, bname="p2")
            resout = "add" >> start + concd1(p1, p2)
            out = av(resout, 3, "out")

            @test nout(resout) == 8

            # Propagates to input of first vertex of p2, input of out and start
            # via p1 and resout as well as to input of split
            @test Δnout!(split, -1)
            @test nin(out) == [nout(start)] == nin(split) == [8]
            @test reduce(vcat, nin.(outputs(split))) == [3, 3]

            # Should basically undo the previous mutation
            @test Δnin(out, +1)
            @test nin(out) == [nout(start)] == nin(split) == [9]
        end

        @testset "Transparent fork block" begin
            start = av(inputvertex("in", 3), 4, "start")
            p1 = iv(start, name="p1")
            p2 = iv(start, name="p2")
            joined = concd1(p1, p2, name="join")
            out = av(joined, 3, "out")

            @test nout(joined) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the conc before resout.
            @test Δnout!(start, -1)
            @test nin(out) == [2nout(start)] == [6]

            # Should basically undo the previous mutation
            @test minΔninfactor(out) == 2
            @test Δnin(out, 2)
            @test nin(out) == [2nout(start)] == [8]
        end

        @testset "Transparent residual fork block" begin
            start = av(inputvertex("in", 3), 8, "start")
            split = av(start, 4, "split")
            p1 = iv(split, name="p1")
            p2 = iv(split, name="p2")
            resout = "add" >> start + concd1(p1, p2, name="join")
            out = av(resout, 3, "out")

            @test nout(resout) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the conc before resout.
            @test Δnout!(split, -1)
            @test nin(out) == [nout(start)] == nin(split) == [6]

            # Should basically undo the previous mutation
            @test Δnin(out, +2)
            @test nin(out) == [nout(start)] == nin(split) == [8]

            @test minΔninfactor(out) == 2
            @test Δnout!(start, -2)
            @test nin(out) == [nout(start)] == [6]
            @test nout(split) == 3
        end

        @testset "Transparent residual fork block with single absorbing path" begin
            start = av(inputvertex("in", 3), 8, "start")
            split = av(start, 3, "split")
            p1 = iv(split, name="p1")
            p2 = iv(split, name="p2")
            p3 = av(split, 2, "p3")
            resout = "add" >> start + concd1(p1, p2, p3, name="join")
            out = av(resout, 3, "out")

            @test nout(resout) == 8

            # Evil action: This will propagate to both p1 and p2 which are in
            # turn both input to the conc before resout.
            @test Δnout!(split, -1)
            @test nin(out) == [nout(start)] == nin(split) == [6]

            # Should basically undo the previous mutation
            @test Δnin(out, 2)
            @test nin(out) == [nout(start)] == nin(split) == [8]

            @test minΔninfactor(out) == 2
            @test Δnout!(start, -2)
            @test nin(out) == [nout(start)] == [2*nout(split) + nout(p3)] == [6]
            @test nout(split) == 2
            @test nout(p3) == 2
        end
    end

    @testset "Size Mutation possibilities" begin
        # Helpers
        struct SizeConstraint{T}
            constraint::Int
            w::T
        end
        NaiveNASlib.minΔnoutfactor(c::SizeConstraint) = c.constraint
        NaiveNASlib.minΔninfactor(c::SizeConstraint) = c.constraint
        function NaiveNASlib.compconstraint!(::NaiveNASlib.ScalarSize, s, c::SizeConstraint, data)
            fv_out = JuMP.@variable(data.model, integer=true)
            JuMP.@constraint(data.model, c.constraint * fv_out ==  nout(data.vertex) - data.noutdict[data.vertex])

            ins = filter(vin -> vin in keys(data.noutdict), inputs(data.vertex))
            fv_in = JuMP.@variable(data.model, [1:length(ins)], integer=true)
            JuMP.@constraint(data.model, [i=1:length(ins)], c.constraint * fv_in[i] ==  nout(ins[i]) - data.noutdict[ins[i]])
        end
        NaiveNASlib.Δsizetype(c::SizeConstraint) = NaiveNASlib.Δsizetype(c.w)
        NaiveNASlib.nout(c::SizeConstraint) = nout(c.w)
        NaiveNASlib.nin(c::SizeConstraint) = nin(c.w)
        NaiveNASlib.Δsize!(c::SizeConstraint, insize::AbstractVector{<:Integer}, outsize::Integer) = Δsize!(c.w, insize, outsize)
        av(size, csize, in... ;name = "av") = absorbvertex(SizeConstraint(csize, SizeDummy(collect(nout.(in)), size)), in..., traitdecoration=tf(name))

        @testset "SizeStack multi inputs" begin
            v1 = av(6,3, inpt(3), name="v1")
            v2 = av(7,2, inpt(3), name="v2")
            v3 = av(8,2, inpt(3), name="v3")

            cc1 = cc(v1, v2, name="cc1")
            cc2 = cc(v3, v2, v1, v2, name="cc2")
            @test minΔnoutfactor(cc1) == 6
            @test minΔnoutfactor(cc2) == 12

            # Expect only v1 to change as size change is not compatible with v2
            @test Δnout!(cc1, -3)
            @test nin(cc1) == [nout(v1), nout(v2)] == [3, 7]
            @test nin(cc2) == [nout(v3), nout(v2), nout(v1), nout(v2)] == [8, 7, 3, 7]
            @test nout(cc1) == sum(nin(cc1))
            @test nout(cc2) == sum(nin(cc2))

            # v1 can't change as it is too small already
            @test Δnout!(cc2, -6)
            @test nin(cc1) == [nout(v1), nout(v2)] == [3, 5]
            @test nin(cc2) == [nout(v3), nout(v2), nout(v1), nout(v2)] == [6, 5, 3, 5]
            @test nout(cc1) == sum(nin(cc1))
            @test nout(cc2) == sum(nin(cc2))

            # Evil action! Must have understanding that change will propagate to cc1 from
            # both v1 and v2
            @test Δnout!(cc2, 9)
            @test nin(cc1) == [nout(v1), nout(v2)] == [6, 7]
            @test nin(cc2) == [nout(v3), nout(v2), nout(v1), nout(v2)] == [8, 7, 6, 7]
            @test nout(cc1) == sum(nin(cc1))
            @test nout(cc2) == sum(nin(cc2))

        end

        @testset "Stacked SizeStacks" begin
            v1 = av(100,1, inpt(3), name="v1")
            v2 = av(100,2, inpt(3), name="v2")
            v3 = av(100,3, inpt(3), name="v3")
            v4 = av(100,5, inpt(3), name="v4")

            cc1 = cc(v2, v3, name="cc1")
            cc2 = cc(cc1, v1, name="cc2")
            cc3 = cc(cc2, v4, v2, name="cc3")
            @test minΔnoutfactor(cc1) == minΔninfactor(cc1) == 6
            @test minΔnoutfactor(cc2) == minΔninfactor(cc2)== 6
            @test minΔnoutfactor(cc3) == minΔninfactor(cc3) == 60 # v2 is input twice through sc2->cc1

            @test Δnout!(cc3, -60)
            @test nin(cc1) == nout.(inputs(cc1)) == [86, 91]
            @test nin(cc2) == nout.(inputs(cc2)) == [177, 87]
            @test nin(cc3) == nout.(inputs(cc3)) == [264, 90, 86]
            @test nout(cc1) == nout(v2) + nout(v3) == 177
            @test nout(cc2) == nout(cc1) + nout(v1) == 264
            @test nout(cc3) == nout(cc2) + nout(v4) + nout(v2) == 440

            v5 = av(10, 3, cc3, name="v5")
            @test minΔnoutfactor(cc1) == minΔninfactor(cc1) == 6
            @test minΔnoutfactor(cc2) == minΔninfactor(cc2)== 6
            @test minΔnoutfactor(cc3) == minΔninfactor(cc3) == 60

            @test Δnout!(v1, 3)
            @test nin(cc1) == nout.(inputs(cc1)) == [86, 91]
            @test nin(cc2) == nout.(inputs(cc2)) == [177, 90]
            @test nin(cc3) == nout.(inputs(cc3)) == [267, 90, 86]
            @test nout(cc1) == nout(v2) + nout(v3) == 177
            @test nout(cc2) == nout(cc1) + nout(v1) == 267
            @test [nout(cc3)] == [(nout(cc2) + nout(v4) + nout(v2))] == nin(v5) == [443]

            # Evil action! Must have understanding that the change in v2 will propagate
            # to cc3 input 1 through cc1 and cc2
            @test Δnout!(v2, -12)
            @test nin(cc1) == nout.(inputs(cc1)) == [74, 103]
            @test nin(cc2) == nout.(inputs(cc2)) == [177, 90]
            @test nin(cc3) == nout.(inputs(cc3)) == [267, 90, 74]
            @test nout(cc1) == nout(v2) + nout(v3) == 177
            @test nout(cc2) == nout(cc1) + nout(v1) == 267
            @test [nout(cc3)] == [(nout(cc2) + nout(v4) + nout(v2))] == nin(v5) == [431]
        end

        @testset "SizeInvariant multi input" begin
            v1 = av(10,1, inpt(3), name="v1")

            ea1 = ea(v1, name="ea1")
            ea2 = ea(ea1, v1, name="ea2")

            @test Δnout!(ea2, -2)

            @test nout(ea2) == nout(ea1) == nout(v1) == 8
            @test nin(ea2) == [nout(ea1), nout(v1)] == [8, 8]
            @test nin(ea1) == [nout(ea1)] == [8]
        end

        @testset "SizeInvariant multi SizeStack input" begin
            v1 = av(7,1, inpt(3), name="v1")
            v2 = av(13,1, inpt(3), name="v2")

            cc1 = cc(v1, v2, name="cc1")
            cc2 = cc(v1, v2, name="cc2")

            ea1 = ea(cc1, cc2, name="ea1")
            ea2 = ea(cc1, cc2, name="ea2")
            ea3 = ea(ea1, ea2, name="ea3")

            @test Δnout!(ea3, -4)

            @test nout(ea3) == nout(ea2) == nout(ea1) == nout(cc2) == nout(cc1) == nout(v2) + nout(v1) == 16
            @test nin(ea3) == nin(ea2) == nin(ea3) == [nout(cc1), nout(cc2)] == [16, 16]
            @test nin(cc2) == nin(cc1) == [nout(v1), nout(v2)] == [6, 10]
        end

        @testset "Stacked SizeInvariant" begin
            v1 = av(100,1, inpt(3), name="v1")
            v2 = av(100,2, inpt(3), name="v2")
            v3 = av(100,3, inpt(3), name="v3")
            v4 = av(100,5, inpt(3), name="v4")

            ea1 = ea(v2, v3, name="ea1")
            ea2 = ea(ea1, v1, name="ea2")
            ea3 = ea(ea2, v4, v2, name="ea3")

            # Everything touches everything in this setup
            @test minΔnoutfactor(ea1) == minΔninfactor(ea1) == 1*2*3*5
            @test minΔnoutfactor(ea2) == minΔninfactor(ea2) == 1*2*3*5
            @test minΔnoutfactor(ea3) == minΔninfactor(ea3) == 1*2*3*5

            @test Δnout!(ea3, -30)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 70
            @test nout(ea1) == nout(ea2) == nout(ea3) == 70

            v5 = av(10, 3, ea3, name="v5")

            @test Δnout!(v1, 60)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 130
            @test [nout(ea1)] == [nout(ea2)] == [nout(ea3)] == nin(v5) == [130]

            # Evil action! Must have understanding that the change in v2 will propagate
            # to ea3 input 1 through ea1 and ea2 and hold off updating it through input 3
            @test Δnout!(v2, -30)
            @test nout(v1) == nout(v2) == nout(v3) == nout(v4) == 100
            @test [nout(ea1)] == [nout(ea2)] == [nout(ea3)] == nin(v5) == [100]
        end

        @testset "Stacked input to SizeInvariant" begin
            v0 = inpt(3, "in")
            v1 = av(20, 2, v0, name="v1")
            v2 = av(30, 3, v0, name="v2")
            v3 = cc(v1,v2,v1, name="v3")
            v4 = av(70, 5, v0, name="v4")
            v5 = ea(v4,v3, name="v5")
            v6 = av(10, 7, v5, name="v6")

            # Evilness: Invariant vertex must change all its inputs and therefore it must take their minΔnoutfactors into account when computing minΔninfactor.
            @test minΔnoutfactor(v4) == 2*2*3*5*7

            @test Δnout!(v5, 2*2*3*5*7)

            @test nout(v5) == nout(v4) == nout(v3) == 490
            @test nin(v5) == [nout(v4), nout(v3)] == [490, 490]
            @test nin(v3) == [nout(v1), nout(v2), nout(v1)] == [140, 210, 140]
        end

        @testset "Invariant input to SizeInvariant" begin
            v0 = inpt(3, "in")
            v1 = av(20, 2, v0, name="v1")
            v2 = av(20, 3, v0, name="v2")
            v3 = ea(v1,v2, name="v3")
            v4 = av(20, 5, v0, name="v4")
            v5 = ea(v4,v3, name="v5")
            v6 = av(10, 7, v5, name="v6")

            # Evilness: Infinite recursion without memoization
            @test minΔnoutfactor(v4) == 2*3*5*7

            @test Δnout!(v5, 2*3*5*7)

            @test nout(v5) == nout(v4) == nout(v3) == 230
            @test nin(v5) == [nout(v4), nout(v3)] == nin(v3) == [nout(v1), nout(v2)] == [230, 230]
        end

        @testset "SizeStack duplicate through SizeInvariant" begin
            v0 = inpt(5, "in")
            v1 = av(2, 2, v0, name="v1")
            v2 = ea(v1, name="v2")
            v3 = ea(v1, name="v3")
            v4 = cc(v2,v3, name="v4")
            v5 = ea(v4, name="v5")
            v6 = cc(v5, v2, name="v6")

            @test minΔnoutfactor(v6) == 3*2
            @test Δnout!(v6, 12)

            @test nout(v6) == sum(nin(v6)) == 18
            @test nout(v5) == nin(v5)[] == 12
            @test nout(v4) == sum(nin(v4)) == 12
            @test nout(v1) == 6
        end

        @testset "SizeInvariant zig-zag" begin
                v0 = inpt(5, "in")
                v1 = av(2, 2, v0, name="v1")
                v2 = av(3, 3, v0, name="v2")
                v3 = cc(v1,v2, name="v3")
                function zigzag(vin1, sc, vin2=v0;name="zig")
                    vnew = av(nout(vin1), sc, vin2, name=name*"_new")
                    vout = ea(vnew, name=name*"_ivB")
                    vcon = ea(vout, vin1, name=name*"_ivA")
                    return vout
                end
                v4 = zigzag(v3, 5, name="z1")
                v5 = zigzag(v4, 7, name="z2")
                v6 = av(11, 11, v5, name="v6")

                expectedΔf = 2*3*5*7*11
                @test minΔnoutfactor(v1) == expectedΔf / 3 # 3 is size constraint for v2
                @test minΔnoutfactor(v2) == expectedΔf / 2 # 2 is size constraint for v1
                @test minΔnoutfactor(v3) == expectedΔf

                @test Δnout!(v2, expectedΔf)
                @test nout(v3) == nout(v4) == nout(v5) == expectedΔf + 2+3
                @test nin(v4) == nout.(inputs(v4)) == [expectedΔf + 2+3]
                @test nin(v5) == nout.(inputs(v5)) == [expectedΔf + 2+3]
        end

        @testset "SizeStack duplicate SizeInvariant mini-zig-zag" begin
            v0 = inpt(5, "in")
            v1 = av(2, 2, v0, name="v1")
            v2 = av(2, 3, v0, name="v2")
            v3 = ea(v1, name="v3")
            v4 = ea(v3, v2, name="v4")
            v5 = av(3, 5, v3, name="v5")
            v6 = cc(v4,v4, name="v6")
            v7 = ea(v6, v6, name="v7")
            v8 = av(4, 7, v7, name="v8")

            expectedΔf = 2*2*3*5*7
            @test minΔnoutfactor(v6) == expectedΔf

            @test Δnout!(v6, expectedΔf)
            @test nin(v8) == [nout(v7)] == [nout(v6)] == [4+expectedΔf]

            @test nin(v6) == [2 + expectedΔf ÷ 2, 2 + expectedΔf ÷ 2]
            @test nout(v1) == nout(v2) == 2 + expectedΔf ÷ 2
            @test nin(v5) == [2 + expectedΔf ÷ 2]
        end

        @testset "Deep SizeStack" begin
            v0 = inpt(3, "in")
            v1 = av(8, 1, v0, name="v1")
            v2 = av(4, 1, v0, name="v2")
            v3 = cc(v1,v2, name= "v3")
            pa1 = ea(v3, name="pa1")
            pb1 = ea(v3, name="pb1")
            pc1 = ea(v3, name="pc1")
            pd1 = av(5, 1, v3, name="pd1")
            pa1pa1 = ea(pa1, name="pa1pa1")
            pa1pb1 = ea(pa1, name="pa1pb1")
            pa2 = cc(pa1pa1, pa1pb1, name = "pa2")
            v4 = cc(pa2, pb1, pc1, pd1, name = "v4")

            @test minΔnoutfactor(v4) == 4

            @test Δnout!(v4, 8)

            @test nout(v4) == 61
            @test nin(v4) == nout.(inputs(v4)) == [28, 14, 14, 5]
            @test nin(pa2) == nout.(inputs(pa2)) == [14, 14]
            @test nin(pa1pa1) == nin(pa1pb1) == [14]
            @test nin(pa1) == nin(pb1) == nin(pc1) == [nout(v3)] == [14]
            @test nin(v3) == [10, 4]
        end

        @testset "Fail invalid size change" begin
            import NaiveNASlib: ΔSizeFailError
            v0 = inpt(3)
            v1 = av(3,3, v0, name="v1")
            v2 = av(5,2, v1, name="v2")

            @test_throws ΔSizeFailError newsizes(ΔNoutExact(v1, 2; fallback=ThrowΔSizeFailError("")), all_in_graph(v1))
            @test_throws ΔSizeFailError newsizes(ΔNoutExact(v1, 3; fallback=ThrowΔSizeFailError("")), all_in_graph(v1))

            @test_throws ΔSizeFailError newsizes(ΔNinExact(v2, 2; fallback=ThrowΔSizeFailError("")), all_in_graph(v1))
            @test_throws ΔSizeFailError newsizes(ΔNinExact(v2, 3; fallback=ThrowΔSizeFailError("")), all_in_graph(v1))

            @test newsizes(ΔNoutExact(v1, 2; fallback=ΔSizeFailNoOp()), all_in_graph(v1))[1] == false
            @test newsizes(ΔNinExact(v2, 2; fallback=ΔSizeFailNoOp()), all_in_graph(v2))[1] == false

            @test (@test_logs (:info, "Giving up") newsizes(ΔNoutExact(v1, 2; fallback=LogΔSizeExec("Giving up")), all_in_graph(v1)))[1] == false

            @test_logs (:warn, r"Could not change nout of .* by 2! Relaxing constraints...") Δnout!(v1, 2)
            @test [nout(v1)] == nin(v2) == [9]

            @test_logs (:warn, r"Could not change nin of .* by -2! Relaxing constraints...") Δnin(v2, -2)
            @test [nout(v1)] == nin(v2) == [3]

            v3 = cc(v1,v2, name="v3")

            @test_logs (:warn, r"Could not change nin of .* by 2 and 6! Relaxing constraints...") Δnin(v3, 2, 6)
            @test [nout(v1)] == nin(v2) == [9]
            @test nin(v3) == [nout(v1), nout(v2)] == [9, 11]

            @test_logs (:warn, r"Could not change nout of .* by -3! Relaxing constraints...") Δnout!(v3, -3)
            @test [nout(v1)] == nin(v2) == [9]
            @test nin(v3) == [nout(v1), nout(v2)] == [9, 9]
        end

    end

    @testset "SizeChangeLogger" begin


        traitfun(name) = t -> SizeChangeLogger(NameInfoStr(), NamedTrait(t, name))
        av(in, size ;name = "av") = absorbvertex(SizeDummy(nout(in), size), in; traitdecoration=traitfun(name))

        @testset "Log size change" begin
            v1 = inpt(3, "v1")
            v2 = av(v1, 10, name="v2")
            v3 = av(v2, 4, name="v3")

            @test @test_logs (:info, "Change nin of v3 by 3") (:info, "Change nout of v2 by 3") Δnin(v3, 3)

            @test @test_logs (:info, "Change nout of v2 by -3") (:info, "Change nin of v3 by -3") Δnout!(v2, -3)

            v4 = av(v1, nout(v3), name="v4")
            v5 = traitconf(traitfun("v5")) >> v3 + v4

            @test @test_logs (:info, "Change nin of v5 by 30 and 30") (:info, "Change nout of v5 by 30") (:info, "Change nout of v3 by 30") (:info, "Change nout of v4 by 30") Δnout!(v5, 30)

            @test @test_logs (:info, "Change nin of v5 by [1,…, 34, -1×10] and [1,…, 34, -1×10]") (:info, "Change nout of v5 by [1,…, 34, -1×10]") (:info, "Change nout of v3 by [1,…, 34, -1×10]") (:info, "Change nout of v4 by [1,…, 34, -1×10]") Δsize!(NaiveNASlib.NeuronIndices(), ΔNoutExact(v5, 10), v5)
        end

        @testset "Clone" begin
            slg = SizeChangeLogger(NameInfoStr(), NamedTrait(SizeAbsorb(), "test"))

            @test slg == clone(slg)

            rename(x;cf=rename) = clone(x, cf=cf)
            rename(x::String; cf) = "success"

            @test name(rename(slg), nothing) == "success"
        end
    end
end
