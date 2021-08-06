# NaiveNASlib

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://USER_NAME.github.io/PACKAGE_NAME.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://USER_NAME.github.io/PACKAGE_NAME.jl/dev)
[![Build status](https://github.com/DrChainsaw/NaiveNASlib.jl/workflows/CI/badge.svg?branch=master)](https://github.com/DrChainsaw/NaiveNASlib.jl/actions)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/DrChainsaw/NaiveNASlib.jl?svg=true)](https://ci.appveyor.com/project/DrChainsaw/NaiveNASlib-jl)
[![Codecov](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DrChainsaw/NaiveNASlib.jl)

NaiveNASlib is a library of functions for mutating computation graphs. It is designed with Neural Architecture Search (NAS) in mind, but can be used for any purpose where doing changes to a model architecture is desired.

It is "batteries excluded" in the sense that it is independent of both neural network implementation and search policy implementation. If you need batteries, check out [NaiveNASflux](https://github.com/DrChainsaw/NaiveNASflux.jl).

Its only contribution to this world is some help with the sometimes annoyingly complex procedure of changing an existing neural network into a new, similar yet different, neural network.

## Basic usage

```julia
]add NaiveNASlib
```

## Contributing

All contributions are welcome. Please file an issue before creating a PR.
