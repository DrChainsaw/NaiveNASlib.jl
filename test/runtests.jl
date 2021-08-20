using Test, NaiveNASlib, NaiveNASlib.Advanced, NaiveNASlib.Extend

include("testutil.jl")

# import Documenter makes include("tests/runtests.jl") workflows cumbersome as NaiveNASlib does no depend on it
# here is my workaround: 
# ]activate test
# dev NaiveNASlib
# include("tests/runtests.jl")
# Delete test/Manifest.toml afterwards or else ]test from main project might not work 

@testset "NaiveNASlib.jl" begin

	@info "Testing computation"
	include("vertex.jl")
	include("compgraph.jl")
	
	include("prettyprint.jl")

	@info "Testing mutation"

	include("mutation/vertex.jl")
	include("mutation/graph.jl")

	@info "Testing size mutation"
	include("mutation/size.jl")

	@info "Testing index mutation"
	include("mutation/select.jl")

	@info "Testing structural mutation"
	include("mutation/structure.jl")

	@info "Testing api"
	include("api/vertex.jl")

	@info "Testing doc examples"
	include("examples.jl")

	if Int !== Int32
		# Don't test documentation unless 64-bit os since some example print numerical types
		import Documenter
		Documenter.doctest(NaiveNASlib)
	end
end
