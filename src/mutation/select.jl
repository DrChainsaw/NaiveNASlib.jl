
function select_inputs end
function select_outputs end

function select_inputs(v::AbstractVertex, inputs::AbstractArray{<:Integer, 1}...)
    select_inputs(base(v), inputs...)
end

function select_inputs(v::CompVertex, inputs::AbstractArray{<:Integer, 1}...)
    select_inputs(v.computation, inputs...)
end

function select_inputs(v::AbsorbVertex)
    select_inputs(v, v.meta)
end

function select_inputs(v::AbstractVertex, m::IoIndices)
    select_inputs(v, m.in...)
end

function select_outputs(v::AbstractVertex, outputs::AbstractArray{<:Integer, 1})
    select_outputs(base(v), outputs)
end

function select_outputs(v::CompVertex, outputs::AbstractArray{<:Integer, 1})
    select_outputs(v.computation, outputs)
end

function select_outputs(v::AbsorbVertex)
    select_outputs(v, v.meta)
end

function select_outputs(v::AbstractVertex, m::IoIndices)
    select_outputs(v, m.out)
end

function select_params(v::AbsorbVertex)
    select_inputs(v)
    select_outputs(v)
end
