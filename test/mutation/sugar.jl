import NaiveNASlib

@testset "Sugar" begin

    struct ScaleByTwo
        f
    end
    (s::ScaleByTwo)(x...) = 2s.f(x...)

    @testset "Create input" begin
        @test issame(inputvertex("input", 5), InputSizeVertex("input", 5))
    end

    @testset "Create immutable" begin
        v = immutablevertex(x -> x * [1 2 3; 4 5 6], 2, inputvertex("in", 3))
        @test nin(v) == [3]
        @test nout(v) == 2
        @test v([1 2]) == [9  12  15]
        v = immutablevertex(identity, 1, inputvertex("in", 1), mutation=IoSize, traitdecoration = t -> NamedTrait(t, "v"))
        @test name(v) == "v"
    end

    @testset "Create absorbing" begin
        v = absorbvertex(x -> x * [1 2 3; 4 5 6], 2, inputvertex("in", 3))
        @test nin(v) == [3]
        @test nout(v) == 2
        @test v([1 2]) == [9  12  15]
        v = absorbvertex(identity, 1, inputvertex("in", 1), mutation=IoSize, traitdecoration = t -> NamedTrait(t, "v"))
        @test name(v) == "v"
    end

    @testset "Create invariant" begin
        v = invariantvertex(x -> 2 .* x, inputvertex("in", 2))
        @test nin(v) == [2]
        @test nout(v) == 2
        @test v([1 2]) == [2 4]
        v = invariantvertex(identity, inputvertex("in", 1), mutation=IoSize, traitdecoration = t -> NamedTrait(t, "v"))
        @test name(v) == "v"
    end

    @testset "Create stacking" begin
        v = conc(inputvertex.(("in1", "in2"), (1,2))..., dims=1, traitdecoration = t -> NamedTrait(t, "v"), outwrap = ScaleByTwo)
        @test nin(v) == [1 ,2]
        @test nout(v) == 3
        @test v(1, [2, 3]) == [2,4,6] # Due to outwrap = ScaleByTwo
        @test name(v) == "v"
        @test typeof(v.base.base.computation) == ScaleByTwo
    end

    @testset "Create element wise" begin

        conf = traitconf(t -> NamedTrait(t, "v"))

        @testset "Create elemwise addition" begin
            v = conf >> inputvertex("in1", 2) + inputvertex("in2", 2) + inputvertex("in3" ,2)

            @test nin(v) == [2, 2, 2]
            @test nout(v) == 2
            @test v([1 2], [3 4], [5 6]) == [9 12]
            @test v([1, 2], [3, 4], [5, 6]) == [9, 12]
            @test v([2], [3, 4], [5, 6]) == [10, 12]
            @test name.(inputs(v)) == ["in1", "in2", "in3"]

            @test_throws DimensionMismatch inputvertex("in1", 2) + inputvertex("in2", 3)

            v = inputvertex("in1", 2) + inputvertex("in2", 2)
            @test v([1, 2], [3, 4]) == [4, 6]

            v = outwrapconf(f -> (x...) -> 2f((x...))) >> inputvertex("in1", 2) + inputvertex("in2", 2)
            @test v([1, 2], [3, 4]) == [8, 12]
        end

        @testset "Create elemwise multiplication" begin
            v = conf >> inputvertex("in1", 2) * inputvertex("in2", 2) * inputvertex("in3" ,2)

            @test nin(v) == [2, 2, 2]
            @test nout(v) == 2
            @test v([1 2], [3 4], [5 6]) == [15 48]
            @test v([1, 2], [3, 4], [5, 6]) == [15, 48]
            @test v([2], [3,4], [5,6]) == [30, 48]
            @test name.(inputs(v)) == ["in1", "in2", "in3"]

            @test_throws DimensionMismatch inputvertex("in1", 2) * inputvertex("in2", 3)

            v = inputvertex("in1", 2) * inputvertex("in2", 2)
            @test v([1, 2], [3, 4]) == [3, 8]
        end
        @testset "Create elemwise subtraction" begin
            v = conf >> inputvertex("in1", 2) - inputvertex("in2", 2)

            @test nin(v) == [2, 2]
            @test nout(v) == 2
            @test v([1 2], [3 4]) == [-2 -2]
            @test v([1, 2], [3, 4]) == [-2, -2]
            @test v([2], [3,4]) == [-1, -2]
            @test name.(inputs(v)) == ["in1", "in2"]

            @test_throws DimensionMismatch inputvertex("in1", 2) - inputvertex("in2", 3)

            v = inputvertex("in1", 2) - inputvertex("in2", 2)
            @test v([1, 2], [3, 4]) == [-2, -2]
        end

        @testset "Create unary subtraction" begin
            v = -(conf >> inputvertex("in1", 2))

            @test nin(v) == [nout(v)] == [2]
            @test v([1 2]) == [-1 -2]
            @test v([1, 2]) == [-1, -2]
            @test name.(inputs(v)) == ["in1"]

            v = -inputvertex("in1", 2)
            @test v([1, 2]) == [-1, -2]
        end

        @testset "Create elemwise division" begin
            v = conf >> inputvertex("in1", 2) / inputvertex("in2", 2)

            @test nin(v) == [2, 2]
            @test nout(v) == 2
            @test v([6 8], [2 4]) == [3 2]
            @test v([6, 8], [2, 4]) == [3, 2]
            @test v([12], [3, 4]) == [4, 3]
            @test name.(inputs(v)) == ["in1", "in2"]

            @test_throws DimensionMismatch inputvertex("in1", 2) / inputvertex("in2", 3)

            v = inputvertex("in1", 2) / inputvertex("in2", 2)
            @test v([6, 8], [2, 4]) == [3, 2]
        end

        @testset "Named elementwise" begin
            v = "eladd" >> inputvertex("in1", 2) + inputvertex("in2", 2)

            @test name(v) == "eladd"
            @test nin(v) == [2, 2]
            @test nout(v) == 2

            v = -("unsub" >> inputvertex("in1", 2))
            @test name(v) == "unsub"
            @test nin(v) == [2]
            @test nout(v) == 2
        end

        @testset "Outwrap shortcut" begin
            scale(f) = ScaleByTwo(f)
            v = scale >> inputvertex("in1", 2) + inputvertex("in2", 2)

            @test nin(v) == [2, 2]
            @test nout(v) == 2

            @test v([1, 2], [3, 4]) == [8, 12]
            @test typeof(v.base.base.computation) == ScaleByTwo
        end
    end
end
