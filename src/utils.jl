module Utils

using LinearAlgebra
using Random

export generate_grid, sample_randomly, get_incumbent

"""
    generate_grid(bounds::AbstractMatrix, points_per_dim::Int)
Generate a regular grid of points within the given boundaries.
bounds: d x 2 matrix, each row is [min, max]
"""
function generate_grid(bounds::AbstractMatrix, points_per_dim::Int)
    dims = size(bounds, 1)
    if dims == 0
        return Matrix{Float64}(undef, 0, 0)
    end
    linspaces = [range(bounds[i, 1], stop=bounds[i, 2], length=points_per_dim) for i in 1:dims]
    grid_tuples = Iterators.product(linspaces...)
    n_points = points_per_dim^dims
    grid_matrix = Matrix{Float64}(undef, dims, n_points)
    for (i, p) in enumerate(grid_tuples)
        grid_matrix[:, i] .= collect(p)
    end
    return grid_matrix
end

"""
    sample_randomly(bounds::AbstractMatrix, n_samples::Int)
Randomly sample points within the given bounds.
bounds: d x 2 matrix
"""
function sample_randomly(bounds::AbstractMatrix, n_samples::Int)
    dims = size(bounds, 1)
    if dims == 0 || n_samples == 0
        return Matrix{Float64}(undef, dims, 0)
    end
    samples = Matrix{Float64}(undef, dims, n_samples)
    for i in 1:dims
        min_b, max_b = bounds[i, :]
        samples[i, :] = rand(n_samples) .* (max_b - min_b) .+ min_b
    end
    return samples
end

"""
    get_incumbent(Y::AbstractVector, maximize::Bool)
Obtain the current best observation (incumbent). (Primarily used in BO)
"""
function get_incumbent(Y::AbstractVector, maximize::Bool)
    if isempty(Y)
        return maximize ? -Inf : Inf
    end
    return maximize ? maximum(Y) : minimum(Y)
end

end