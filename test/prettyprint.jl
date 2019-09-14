@testset "Compressed array string" begin
    @test NaiveNASlib.compressed_string([1,2,3,4]) == "[1, 2, 3, 4]"
    @test NaiveNASlib.compressed_string(1:23) == "[1,…, 23]"
    @test NaiveNASlib.compressed_string([1,2,3,5,6,8,9,10:35...]) == "[1, 2, 3, 5, 6, 8,…, 35]"
    @test NaiveNASlib.compressed_string(-ones(Int, 24)) == "[-1×24]"
    @test NaiveNASlib.compressed_string([-1, 5:9...,-ones(Int, 6)...,23,25,99,100,102,23:32...,34]) ==  "[-1, 5,…, 9, -1×6, 23, 25, 99, 100, 102, 23,…, 32, 34]"
    @test NaiveNASlib.compressed_string([1,2,4:13..., 10:20...,-1]) == "[1, 2, 4,…, 13, 10,…, 20, -1]"
end
