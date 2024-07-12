@testset "CompGraph" begin


    av(name, in, outsize) = absorbvertex(name, MatMul(nout(in), outsize), in)

    function testgraph() 
        iv1 = inputvertex("iv1", 1)
        iv2 = inputvertex("iv2", 2)

        v1 = av("v1", iv1, 3)
        v2 = av("v2", v1, 4)

        v3 = av("v3", iv2, 5)
        v4 = av("v4", v1, 5)

        v5 = "v5" >> v3 + v4

        v6 = conc("v6", v2, v3; dims=1)

        CompGraph([iv1, iv2], [v6, v5])
    end

    @testset "summarytable" begin
        import NaiveNASlib: summarytable

        g = testgraph()
        t = summarytable(g, "vname"=>name, nin, nout)

        vs = vertices(g)
        @testset "Check vertex $i" for i in eachindex(t[1]) 
            v = vs[t[2][i]]
            @test name(v) == t.vname[i]
            @test name.(inputs(v)) == name.(vs[t[3][i]])
            @test nin(v) == t.Nin[i]
            @test nout(v) == t.Nout[i]
        end
    end

    @testset "graphsummary" begin
        g = testgraph()
        str = sprint((args...) -> graphsummary(args..., "vname"=>name, nin, nout; highlighters=tuple()), g)

        expnames = name.(vertices(g))
        innames = name.(inputs(g))
        outnames = name.(outputs(g))
        hnames = setdiff(expnames, innames, outnames)

        foundnames = String[]

        for row in split(str, '\n')
            if contains(row, "Input  ")
                m = match(Regex(string('(', join(innames, '|'), ')')), row)
                @test !isnothing(m)
                append!(foundnames, m.captures)
            end

            if contains(row, "Hidden")
                m = match(Regex(string('(', join(hnames, '|'), ')')), row)
                @test !isnothing(m)
                append!(foundnames, m.captures)
            end

            if contains(row, "Output")
                m = match(Regex(string('(', join(outnames, '|'), ')')), row)
                @test !isnothing(m)
                append!(foundnames, m.captures)
            end
        end

        @test sort(expnames) == sort(foundnames)
    end

    @testset "pretty print array" begin
        g = testgraph()

        str = sprint(show, CompGraph[g])

        @test str == "CompGraph[CompGraph($(nvertices(g)) vertices)]"
    end
end


@testset "Compressed array string" begin
    @test NaiveNASlib.compressed_string([1,2,3,4]) == "[1, 2, 3, 4]"
    @test NaiveNASlib.compressed_string(1:23) == "[1,…, 23]"
    @test NaiveNASlib.compressed_string([1,2,3,5,6,8,9,10:35...]) == "[1, 2, 3, 5, 6, 8,…, 35]"
    @test NaiveNASlib.compressed_string(-ones(Int, 24)) == "[-1×24]"
    @test NaiveNASlib.compressed_string([-1, 5:9...,-ones(Int, 6)...,23,25,99,100,102,23:32...,34]) ==  "[-1, 5,…, 9, -1×6, 23, 25, 99, 100, 102, 23,…, 32, 34]"
    @test NaiveNASlib.compressed_string([1,2,4:13..., 10:20...,-1]) == "[1, 2, 4,…, 13, 10,…, 20, -1]"
end
