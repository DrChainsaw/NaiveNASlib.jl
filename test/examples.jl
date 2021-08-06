
@testset "Examples" begin
    using Markdown: @md_str

    @testset "Tutorials" begin
        include("examples/quicktutorial.jl")
        include("examples/advancedtutorial.jl")
    end

end
