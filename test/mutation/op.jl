import NaiveNASlib
import NaiveNASlib:reset_in!, reset_out!, reset!
import InteractiveUtils:subtypes

@testset "Vertex mutation operations" begin

    expectedtype(t::Type{<:MutationOp}) = Integer
    expectedtype(t::Type{IoIndices}) = AbstractArray{<:Integer,1}

    @testset "Method contracts $subtype" for subtype in implementations(MutationOp)
        @test hasmethod(Δnin, (subtype,expectedtype(subtype)))
        @test hasmethod(Δnout, (subtype, expectedtype(subtype)))
        @test hasmethod(clone, (subtype,))
    end

    @testset "Method contracts $subtype" for subtype in implementations(MutationState)
        @test hasmethod(nin, (subtype,))
        @test hasmethod(nout, (subtype,))
        @test hasmethod(in_inds, (subtype,))
        @test hasmethod(out_inds, (subtype,))
    end

    @testset "IoSize" begin
        size = IoSize(3, 4)
        @test nin(size) == [3]
        @test nout(size) == 4

        Δnin(size, -1)
        @test nin(size) == [2]
        @test nout(size) == 4

        Δnout(size, 2)
        @test nin(size) == [2]
        @test nout(size) == 6

        @test in_inds(size) == [1:2]
        @test out_inds(size) == 1:6

        size = IoSize([2, 3, 4], 5)
        @test nin(size) == [2, 3, 4]

        Δnin(size, 1,-2, 3)
        @test nin(size) == [3, 1 ,7]

        Δnin(size, missing,4, -2)
        @test nin(size) == [3, 5 ,5]

        @test in_inds(size) == [1:3, 1:5, 1:5]

        @test issame(size, clone(size))
    end

    @testset "IoIndices" begin
        inds = IoIndices([3,4], 5)

        @test nin(inds) == [3, 4]
        @test nout(inds) == 5

        Δnin(inds, [1,2], [2,3,4])
        @test nin(inds) == [2, 3]
        @test nout(inds) == 5
        @test in_inds(inds) == [[1, 2], [2,3,4]]

        reset_in!(inds)
        @test in_inds(inds) == [[1, 2], [1,2,3]]

        Δnout(inds, [2,4])
        @test nin(inds) == [2, 3]
        @test nout(inds) == 2
        @test out_inds(inds) == [2, 4]

        reset_out!(inds)
        @test out_inds(inds) == [1, 2]

        Δnin(inds, missing, [-1, 2, -1])
        @test in_inds(inds) == [[1, 2], [-1,2,-1]]

        @test issame(inds, clone(inds))
    end

    @testset "IoChange" begin
        s = IoChange(3, 4)

        @test nin(s) == [3]
        @test nout(s) == 4
        @test in_inds(s) == [1:3]
        @test out_inds(s) == 1:4

        Δnin(s, -1)
        @test nin(s) == [2]
        @test nout(s) == 4
        @test in_inds(s) == [1:2]

        s = IoChange([3,4], 5)

        @test nin(s) == [3, 4]
        @test nout(s) == 5
        @test in_inds(s) == [1:3, 1:4]
        @test out_inds(s) == 1:5

        Δnin(s, 1, -2)
        @test nin(s) == [4, 2]
        @test nout(s) == 5
        @test in_inds(s) == [vcat(1:3, -1), 1:2]
        @test out_inds(s) == 1:5

        Δnout(s, 2)
        @test nin(s) == [4, 2]
        @test nout(s) == 7
        @test in_inds(s) == [vcat(1:3, -1), 1:2]
        @test out_inds(s) == vcat(1:5, [-1, -1])

        Δnin(s, [1, -1, 2, -1], [2, -1])
        @test nin(s) == [4, 2]
        @test nout(s) == 7
        @test in_inds(s) == [[1, -1, 2, -1], [2, -1]]
        @test out_inds(s) == vcat(1:5, [-1, -1])

        Δnout(s, [-1, 2, 3, -1, -1, 5, 7])
        @test nin(s) == [4, 2]
        @test nout(s) == 7
        @test in_inds(s) == [[1, -1, 2, -1], [2, -1]]
        @test out_inds(s) == [-1, 2, 3, -1, -1, 5, 7]

        Δnin(s, [1, 3], [-1, 2, 1])
        @test nin(s) == [2, 3]
        @test in_inds(s) == [[1, 3], [-1, 2, 1]]

        Δnout(s, [2, -1, 3, -1])
        @test nout(s) == 4
        @test out_inds(s) == [2, -1, 3, -1]

        reset_in!(s)
        @test nin_org(s) == nin(s) == [2, 3]
        @test in_inds(s) == [1:2, 1:3]
        @test nout_org(s) != nout(s)
        @test out_inds(s) == [2, -1, 3, -1]

        reset_out!(s)
        @test nout_org(s) == nout(s) == 4
        @test out_inds(s) == 1:4

        Δnin(s, 2, -1)
        @test nin(s) == [4, 2]

        Δnin(s, missing, [1, 2])
        @test nin(s) == [4, 2]

        s = IoChange([4 ,2], 4)
        Δnin(s, -1, 3)
        reset_in!(s)

        @test in_inds(s) == [[1,2,3], [1,2,3,4,5]]

        Δnout(s, 3)
        reset_out!(s)
        @test out_inds(s) == 1:7

    end
end
