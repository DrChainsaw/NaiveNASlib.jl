"""
    mutate_inputs(v::MutationVertex)

Apply the input size mutation from the given vertex.

In the context of a neural network, this means removing or adding columns or rows from a matrix
"""
function mutate_inputs end

"""
    mutate_outputs(v::MutationVertex)

Apply the output size mutation from the given vertex.

In the context of a neural network, this means removing or adding columns or rows from a matrix
"""
function mutate_outputs end

"""
    apply_mutation(v::AbstractVertex)

Apply the size mutation from the given vertex.

In the context of a neural network, this means removing or adding columns or rows from a matrix
"""
function apply_mutation(v::AbstractVertex) end
function apply_mutation(v::MutationVertex)
    mutate_inputs(v)
    mutate_outputs(v)
end
apply_mutation(g::CompGraph) = apply_mutation.(unique(mapfoldl(flatten, vcat, g.outputs)))

## Input mutation. Basically just boilerplate-y traversal of the vertex composition hierachy
# until we hit the computation to mutate, then someone else will do the actual work
mutate_inputs(v::CompVertex, inputs...) = mutate_inputs(v.computation, inputs...)
mutate_inputs(v::AbstractVertex, inputs...) = mutate_inputs(base(v), inputs...)
mutate_inputs(v::AbstractVertex, s::IoIndices) = mutate_inputs(v, s.in...)
mutate_inputs(v::AbstractVertex, s::IoSize) = mutate_inputs(v, nin(s)...)
function mutate_inputs(v::AbstractVertex, s::InvIndices)
    mutate_inputs(v, s.inds)
    mutate_outputs(v, s.inds)
    reset!(s)
 end
function mutate_inputs(v::AbstractVertex, s::InvSize)
    mutate_inputs(v, nin(s))
    mutate_outputs(v, nout(s))
 end
function mutate_inputs(v::MutationVertex)
    mutate_inputs(v, op(v))
    reset_in!(op(v))
end

# Noops
function mutate_inputs(f::Function, inputs...) end #Maybe bad idea as it hides errors
function mutate_inputs(v::AbstractVertex, s::NoOp) end


# Output mutations. Also just traversal of the vertex composition hierachy.
mutate_outputs(v::CompVertex, outputs) = mutate_outputs(v.computation, outputs)
mutate_outputs(v::AbstractVertex, outputs) =mutate_outputs(base(v), outputs)
mutate_outputs(v::AbstractVertex, s::IoIndices) = mutate_outputs(v, s.out)
mutate_outputs(v::AbstractVertex, s::IoSize) = mutate_outputs(v, nout(s))
function mutate_outputs(v::AbstractVertex, s::InvIndices)
    mutate_outputs(v, s.inds)
    mutate_inputs(v, s.inds)
    reset!(s)
 end
function mutate_outputs(v::AbstractVertex, s::InvSize)
    mutate_outputs(v, nout(s))
    mutate_inputs(v, nin(s))
    reset!(s)
end
function mutate_outputs(v::MutationVertex)
    mutate_outputs(v, op(v))
    reset_out!(op(v))
end

# Noops
function mutate_outputs(f::Function, inputs...) end #Maybe bad idea as it hides errors
function mutate_outputs(v::AbstractVertex, s::NoOp) end
