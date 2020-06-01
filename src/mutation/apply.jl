"""
    mutate_inputs(v::MutationVertex; kwargs...)

Apply the input size mutation from the given vertex.

In the context of a neural network, this means removing or adding columns or rows from a matrix.

Supplied `kwargs` will be passed on to computation.
"""
function mutate_inputs end

"""
    mutate_outputs(v::MutationVertex; kwargs...)

Apply the output size mutation from the given vertex.

In the context of a neural network, this means removing or adding columns or rows from a matrix.

Supplied `kwargs` will be passed on to computation.
"""
function mutate_outputs end

"""
    apply_mutation(v::AbstractVertex; kwargs...)

Apply the size mutation from the given vertex.

In the context of a neural network, this means removing or adding columns or rows from a matrix.

Supplied `kwargs` will be passed on to computation.
"""
function apply_mutation(v::AbstractVertex; kwargs...) end
function apply_mutation(v::MutationVertex; kwargs...)
    mutate_inputs(v; kwargs...)
    mutate_outputs(v; kwargs...)
end
apply_mutation(g::CompGraph; kwargs...) = apply_mutation.(vertices(g); kwargs...)

## Input mutation. Basically just boilerplate-y traversal of the vertex composition hierachy
# until we hit the computation to mutate, then someone else will do the actual work
mutate_inputs(v::CompVertex, inputs...; kwargs...) = mutate_inputs(v.computation, inputs...; kwargs...)
mutate_inputs(v::AbstractVertex, inputs...; kwargs...) = mutate_inputs(base(v), inputs...; kwargs...)
mutate_inputs(v::AbstractVertex, s::IoChange; kwargs...) = mutate_inputs(v, in_inds(s)...; kwargs...)
mutate_inputs(v::AbstractVertex, s::IoIndices; kwargs...) = mutate_inputs(v, in_inds(s)...; kwargs...)
mutate_inputs(v::AbstractVertex, s::IoSize; kwargs...) = mutate_inputs(v, nin(s)...; kwargs...)

function mutate_inputs(v::MutationVertex; kwargs...)
    mutate_inputs(v, op(v); kwargs...)
    reset_in!(op(v))
end

# Noops
function mutate_inputs(f::Function, inputs...; kwargs...) end #Maybe bad idea as it hides errors
function mutate_inputs(v::AbstractVertex, s::NoOp; kwargs...) end


# Output mutations. Also just traversal of the vertex composition hierachy.
mutate_outputs(v::CompVertex, outputs; kwargs...) = mutate_outputs(v.computation, outputs; kwargs...)
mutate_outputs(v::AbstractVertex, outputs; kwargs...) = mutate_outputs(base(v), outputs; kwargs...)
mutate_outputs(v::AbstractVertex, s::IoChange; kwargs...) = mutate_outputs(v, out_inds(s); kwargs...)
mutate_outputs(v::AbstractVertex, s::IoIndices; kwargs...) = mutate_outputs(v, out_inds(s); kwargs...)
mutate_outputs(v::AbstractVertex, s::IoSize; kwargs...) = mutate_outputs(v, nout(s); kwargs...)

function mutate_outputs(v::MutationVertex; kwargs...)
    mutate_outputs(v, op(v); kwargs...)
    reset_out!(op(v))
end

# Noops
function mutate_outputs(f::Function, inputs...; kwargs...) end #Maybe bad idea as it hides errors
function mutate_outputs(v::AbstractVertex, s::NoOp; kwargs...) end
