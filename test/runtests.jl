using NaiveNASlib
using Test

@testset "NaiveNASlib.jl" begin

    function implementations(T::Type)
        return mapreduce(t -> isabstracttype(t) ? implementations(t) : t, vcat, subtypes(T), init=[])
    end
    
    @info "Testing computation"

    include("vertex.jl")
    include("compgraph.jl")

    @info "Testing mutation"

    include("mutation/meta.jl")
    include("mutation/vertex.jl")
    include("mutation/select.jl")

end
