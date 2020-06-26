@testset "Apply mutation" begin

    inpt(size, name="in") = InputSizeVertex(name, size)
    nt(name) = t -> NamedTrait(t, name)
    tf(name) = t -> nt(name)(SizeChangeValidation(t))
    av(outsize, in, name="av") = absorbvertex(identity, outsize, in, traitdecoration=tf(name))

    function mcv(nin, nout, in, name="mcv")
        mm = MatMul(nin, nout)
        return absorbvertex(mm, nout, in, traitdecoration=tf(name)), mm
    end

    function NaiveNASlib.Δnin(v::AbstractVertex, inds::Vector{<:Integer}...)
        function valfun(vv)
            vv ∉ inputs(v) && return ones(nout_org(vv))
            tosel = inds[vv .== inputs(v)]
            value = -10 * ones(nout_org(vv))
            selinds = filter(i -> i > 0, inds[inputs(v) .== vv][1])
            value[selinds] .= 1000
            return value
        end
        Δnin(v, (length.(inds) .- nin(v))...)
        Δoutputs(OutSelectRelaxed(), v, valfun)
    end

    function NaiveNASlib.Δnout(v::AbstractVertex, inds::Vector{<:Integer})
        function valfun(vv)
            vv != v && return ones(nout_org(vv))
            value = -10 * ones(nout_org(vv))
            value[filter(i -> i > 0, inds)] .= 1000
            return value
        end
        Δnout(v, length(inds) .- nout(v))
        Δoutputs(OutSelectRelaxed(), v, valfun)
    end

    @testset "AbsorbVertex mutation" begin
        nin1, nout1 = 3, 5
        nin2, nout2 = 5, 2
        iv1 = av(nin1, inpt(nin1), "iv1")
        mmv1, mm1 = mcv(nin1, nout1, iv1, "mmv1")
        mmv2, mm2 = mcv(nin2, nout2, mmv1, "mmv2")

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
    end

    @testset "StackingVertex mutation" begin

        mmv1, mm1 = mcv(2, 3, inpt(2, "in1"), "mmv1")
        mmv2, mm2 = mcv(3, 4, inpt(3, "in2"), "mmv2")
        cc1 = conc(mmv1, mmv2, dims=2, traitdecoration=tf("cc1"))
        mmv3, mm3 = mcv(7, 3, cc1, "mmv3")

        Δnout(cc1, Integer[1,3,5,7])
        apply_mutation.((cc1, mmv1, mmv2, mmv3))

        @test mm1.W == [1 5; 2 6]
        @test mm2.W == [4 10; 5 11; 6 12]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 7 14 21]

        Δnout(mmv2, [1, -1, -1])
        apply_mutation.((cc1, mmv1, mmv2, mmv3))

        @test mm1.W == [1 5; 2 6]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 0 0 0; 0 0 0]

        Δnout(mmv1, [2, -1, -1, -1])
        apply_mutation.((cc1, mmv1, mmv2, mmv3))

        @test mm1.W == [5 0 0 0; 6 0 0 0]
        @test mm2.W == [4 0 0; 5 0 0; 6 0 0]
        @test mm3.W == [3 10 17; 0 0 0; ; 0 0 0; 0 0 0; 5 12 19; 0 0 0; 0 0 0]

        Δnin(mmv3, [1, 5])
        apply_mutation.((cc1, mmv1, mmv2, mmv3))

        @test mm1.W == reshape([5; 6], :, 1)
        @test mm2.W == reshape([4; 5; 6], :, 1)
        @test mm3.W == [3 10 17; 5 12 19]
    end

    @testset "InvariantVertex mutation" begin

        addsize = 7
        mmv1, mm1 = mcv(2, addsize, inpt(2, "in1"), "mmv1")
        mmv2, mm2 = mcv(3, addsize, inpt(3, "in2"), "mmv2")
        add = "add" >> mmv1 + mmv2
        mmv3, mm3 = mcv(addsize, 3, add, "mmv3")

        Δnout(add, Integer[1,3,5,7])
        apply_mutation.((add, mmv1, mmv2, mmv3))

        @test mm1.W == [1 5 9 13; 2 6 10 14]
        @test mm2.W == [1 7 13 19; 2 8 14 20; 3 9 15 21]
        @test mm3.W == [1 8 15; 3 10 17; 5 12 19; 7 14 21]

        Δnout(mmv2, [2, 3, -1, -1, -1])
        apply_mutation.((add, mmv1, mmv2, mmv3))

        @test mm1.W == [5 9 0 0 0; 6 10 0 0 0]
        @test mm2.W == [7 13 0 0 0; 8 14 0 0 0; 9 15 0 0 0]
        @test mm3.W == [3 10 17; 5 12 19; 0 0 0; 0 0 0; 0 0 0]
    end

    @testset "No apply to vertex with no inputs" begin
        mmv1, mm1 = mcv(2, 4, inpt(2, "in"), "mmv1")
        mmv2, mm2 = mcv(4, 3, mmv1, "mmv2")

        w1pre = mm1.W
        w2pre = mm2.W

        remove_edge!(mmv1, mmv2)
        apply_mutation.([mmv1, mmv2])

        @test mm1.W == w1pre
        @test mm2.W == w2pre
    end

    @testset "Size only mutation" begin
        mutable struct SizeProbe
            nin
            nout
        end
        NaiveNASlib.mutate_inputs(p::SizeProbe, newsize...) = p.nin = newsize[1]
        NaiveNASlib.mutate_outputs(p::SizeProbe, newsize) = p.nout = newsize
        NaiveNASlib.minΔninfactor(::SizeProbe) = 1
        NaiveNASlib.minΔnoutfactor(::SizeProbe) = 1

        p = SizeProbe(3,5)
        in = av(3, inpt(1, "in"))
        v = absorbvertex(p, p.nout, in, traitdecoration = tf("v"))

        Δnin(v, -1)
        apply_mutation(v)
        @test p.nin == [1,2]
        @test p.nout == [1,2,3,4,5]

        Δnout(v, 2)
        apply_mutation(v)
        @test p.nin == [1, 2]
        @test p.nout == [1,2,3,4,5,-1,-1]
    end

    @testset "Pass kwargs" begin
        mutable struct KwargsProbe
            kws
        end
        KwargsProbe() = KwargsProbe(nothing)
        NaiveNASlib.mutate_inputs(p::KwargsProbe, newsize...; kwargs...) = p.kws = kwargs
        NaiveNASlib.mutate_outputs(p::KwargsProbe, newsize; kwargs...) = p.kws = kwargs

        compgraph_apply(v; kwargs...) = apply_mutation(CompGraph(inputs(v), [v]); kwargs...)

        @testset "Kwargs pass $f" for f in (
            mutate_inputs,
            mutate_outputs,
            apply_mutation,
            compgraph_apply)
            p = KwargsProbe()
            iv = inputvertex("in", 1)
            v = absorbvertex(p, 1, iv)

            f(v; test=:pass)
            @test p.kws == pairs((test=:pass,))
        end
    end
end
