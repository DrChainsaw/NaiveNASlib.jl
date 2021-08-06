# Size Strategies

As shown in [Advanced Tutorial](@ref) size strategies give a high degree of flexibility when it comes to changing the size of vertices in the graph.

All size strategies are executed through the [`Δsize!`](@ref) function.

Imported to namespace by
```julia
using NaiveNASlib.Advanced
```

```@docs
DefaultJuMPΔSizeStrategy
```

```@docs
ΔNinExact
```

```@docs
ΔNinRelaxed
```

```@docs
ΔNin
```

```@docs
ΔNoutExact
```

```@docs
ΔNoutRelaxed
```

```@docs
ΔNout
```

```@docs
WithUtilityFun
```

```@docs
LogΔSizeExec
```

```@docs
ThrowΔSizeFailError
```

```@docs
ΔSizeFailNoOp
```

```@docs
AlignNinToNout
```

```@docs
TruncateInIndsToValid
```

```@docs
TimeLimitΔSizeStrategy
```

```@docs
TimeOutAction
```

```@docs
AfterΔSizeCallback
```

```@docs
logafterΔsize
```

```@docs
validateafterΔsize
```