using Test, NaiveNASlib, NaiveNASlib.Advanced, NaiveNASlib.Extend, Documenter

include("testutil.jl")

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

	@info "Testing sugar"
	include("mutation/sugar.jl")

	@info "Testing doc examples"
	include("examples.jl")

	if Int !== Int32
		doctest(NaiveNASlib)
	end
end
