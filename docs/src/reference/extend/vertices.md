# Vertex Types

While it is not generally expected that implementations should need to define new vertex types, the existing ones are typically useful for dispatching.

Imported to namespace by
```julia
using NaiveNASlib.Extend
```

```@docs
base(::AbstractVertex)
AbstractVertex
InputVertex
InputSizeVertex
CompVertex
MutationVertex
OutputsVertex
```