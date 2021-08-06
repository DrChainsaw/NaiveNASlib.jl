using Documenter, Literate, NaiveNASlib, NaiveNASlib.Advanced, NaiveNASlib.Extend

const nndir = joinpath(dirname(pathof(NaiveNASlib)), "..")

function literate_example(sourcefile; rootdir=nndir, sourcedir = "test/examples", destdir="docs/src/examples")
    fullpath = Literate.markdown(joinpath(rootdir, sourcedir, sourcefile), joinpath(rootdir, destdir); flavor=Literate.CommonMarkFlavor(), mdstrings=true)
    dirs = splitpath(fullpath)
    srcind = findfirst(==("src"), dirs)
    joinpath(dirs[srcind+1:end]...)
end

quicktutorial = literate_example("quicktutorial.jl")
@show advancedtutorial = literate_example("advancedtutorial.jl")

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
                    ],
                    "Extend" => [
                        "reference/extend/vertices.md",
                        "reference/extend/strategies.md",
                        "reference/extend/traits.md",
                        "reference/extend/misc.md"
                    ]
                ]
            ],
            modules = [NaiveNASlib],
        )

function touchfile(filename, rootdir=nndir, destdir="test/examples")
    filepath = joinpath(rootdir, destdir, filename)
    isfile(filepath) && return
    write(filepath, "md\"\"\" # A Header \"\"\"")
end

deploydocs(
    repo = "github.com/DrChainsaw/NaiveNASlib.jl.git",
)
