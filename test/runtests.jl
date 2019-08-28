using NaiveNASlib
using Test
import JuMP

include("testutil.jl")

@testset "NaiveNASlib.jl" begin

        @testset "Junipertest" begin
                import JuMP
                import Juniper
                import Ipopt
                import Cbc

                optimizer = Juniper.Optimizer
                params = Dict{Symbol,Any}()
                params[:nl_solver] = JuMP.with_optimizer(Ipopt.Optimizer, print_level=0, sb="yes")
                params[:mip_solver] = JuMP.with_optimizer(Cbc.Optimizer, logLevel=0)
                params[:log_levels] = []
                model = JuMP.Model(JuMP.with_optimizer(optimizer, params))

                x = JuMP.@variable(model, x[1:9], Int)
                JuMP.@constraint(model, x .>= 1)

                JuMP.@constraint(model, x[2] == 10)
                JuMP.@constraint(model, x[3] == x[2])
                JuMP.@constraint(model, x[7] == x[2])
                JuMP.@constraint(model, x[4] == 3)
                JuMP.@constraint(model, x[6] + x[8] == x[7])

                xtargets = [9, 8, 8, 3, 3, 5, 8, 3, 4]
                objective = JuMP.@NLexpression(model, objective[i=1:length(x)], (x[i]/xtargets[i] - 1)^2)
                JuMP.@NLobjective(model, Min, sum(objective[i] for i in 1:length(objective)))

                JuMP.optimize!(model)
        end

        @info "Testing computation"

        include("vertex.jl")
        include("compgraph.jl")

        @info "Testing mutation"

        include("mutation/op.jl")
        include("mutation/vertex.jl")

        @info "Testing size mutation"

        include("mutation/size.jl")

        @info "Testing index mutation"

        include("mutation/apply.jl")
        include("mutation/select.jl")

        @info "Testing structural mutation"

        include("mutation/structure.jl")

        @info "Testing sugar"

        include("mutation/sugar.jl")
end
