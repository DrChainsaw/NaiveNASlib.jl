using NaiveNASlib
using Test

function implementations(T::Type)
    return mapreduce(t -> isabstracttype(t) ? implementations(t) : t, vcat, subtypes(T), init=[])
end

@testset "NaiveNASlib.jl" begin

    @info "Testing computation"

    include("vertex.jl")
    include("compgraph.jl")

    @info "Testing mutation"

    include("mutation/meta.jl")
    include("mutation/vertex.jl")
    include("mutation/select.jl")

end
