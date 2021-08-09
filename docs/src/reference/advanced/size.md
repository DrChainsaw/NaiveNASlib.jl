# Size Strategies

As shown in [Advanced Tutorial](@ref) size strategies give a high degree of flexibility when it comes to changing the size of vertices in the graph.

All size strategies are executed through the [`Δsize!`](@ref) function.

Imported to namespace by
```julia
using NaiveNASlib.Advanced
```

```@docs
DefaultJuMPΔSizeStrategy
ΔNinExact
ΔNinRelaxed
ΔNin

ΔNoutExact
ΔNoutRelaxed
ΔNout(args...)

ΔNout
WithUtilityFun
WithKwargs
LogΔSizeExec
ThrowΔSizeFailError
ΔSizeFailNoOp
AlignNinToNout
TruncateInIndsToValid
TimeLimitΔSizeStrategy
TimeOutAction
AfterΔSizeCallback
logafterΔsize
validateafterΔsize
```
