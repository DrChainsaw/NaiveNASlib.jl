using Documenter, Literate, NaiveNASlib, NaiveNASlib.Advanced, NaiveNASlib.Extend

NaiveNASlib.GRAPHSUMMARY_USE_HIGHLIGHTS[] = false

# This is to avoid truncation of outputs in doostrings, primarily for graphsummary
defaultcols = haskey(ENV, "COLUMNS") ? ENV["COLUMNS"] : nothing
ENV["COLUMNS"] = 1000
const nndir = joinpath(dirname(pathof(NaiveNASlib)), "..")

try

    function literate_example(sourcefile; rootdir=nndir, sourcedir = "test/examples", destdir="docs/src/examples", kwargs...)
        fullpath = Literate.markdown(joinpath(rootdir, sourcedir, sourcefile), joinpath(rootdir, destdir); flavor=Literate.DocumenterFlavor(), mdstrings=true, kwargs...)
        dirs = splitpath(fullpath)
        srcind = findfirst(==("src"), dirs)
        joinpath(dirs[srcind+1:end]...)
    end

    quicktutorial = literate_example("quicktutorial.jl")
    advancedtutorial = literate_example("advancedtutorial.jl"; codefence="````julia" => "````")

    makedocs(   sitename="NaiveNASlib",
                root = joinpath(nndir, "docs"), 
                format = Documenter.HTML(
                    prettyurls = get(ENV, "CI", nothing) == "true"
                ),
                pages = [
                    "index.md",
                    quicktutorial,
                    advancedtutorial,
                    "terminology.md",
                    "API Reference" => [
                        "reference/simple/createvertex.md",
                        "reference/simple/graph.md",
                        "reference/simple/queryvertex.md",
                        "reference/simple/mutatevertex.md",
                        "Advanced" => [
                            "reference/advanced/graphquery.md",
                            "reference/advanced/size.md",
                            "reference/advanced/structure.md",
                            "reference/advanced/traits.md",
                            "reference/advanced/infixconf.md",
                        ],
                        "Extend" => [
                            "reference/extend/vertices.md",
                            "reference/extend/strategies.md",
                            "reference/extend/traits.md",
                            "reference/extend/misc.md",
                        ],
                    ],
                    "Internal" => [
                        "reference/internal/internal.md",
                    ],
                ],
                modules = [NaiveNASlib],
                warnonly=[:missing_docs],
            )
    function touchfile(filename, rootdir=nndir, destdir="test/examples")
        filepath = joinpath(rootdir, destdir, filename)
        isfile(filepath) && return
        write(filepath, """
        md\"\"\"
        # Markdown header
        \"\"\"
        """)
    end

    if get(ENV, "CI", nothing) == "true"
        deploydocs(
            repo = "github.com/DrChainsaw/NaiveNASlib.jl.git",
            push_preview=true
        )
    end

finally
    NaiveNASlib.GRAPHSUMMARY_USE_HIGHLIGHTS[] = true
    if isnothing(defaultcols)
        delete!(ENV, "COLUMNS")
    else
        ENV["COLUMNS"] = defaultcols
    end
end

nothing # Just so that include("make.jl") does not return anything