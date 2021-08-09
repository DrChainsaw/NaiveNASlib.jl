# Vertex Mutation

While most mutating functions operate on a single vertex or few for convenience, they typically make other modifications in the graph for it to stay
size consistent. This is possible due to how vertices are able to provide their neighbours.

```@docs
Δnin!
Δnout!
Δsize!(::AbstractΔSizeStrategy)
Δsize!(::CompGraph)
Δsize!(::AbstractVector{<:AbstractVertex}) 
relaxed
insert!
remove!
create_edge!
remove_edge!
```






