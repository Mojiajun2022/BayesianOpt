module Maximizers

using LinearAlgebra
using Random
using Optim
using StatsBase
using GaussianProcesses
using DecisionTree
using ..Utils
using ..AcquisitionFunctions
using ..GPModel: predict_with_gp, GPE
using ..RandomForestModel

export maximize_acquisition, find_max_variance_point, find_max_rf_variance_point, MaximizerType, GRID, RANDOM, LBFGS, PROBSAMPLE

@enum MaximizerType GRID RANDOM LBFGS PROBSAMPLE

const DEFAULT_GRID_POINTS = 20
const DEFAULT_RANDOM_SAMPLES = 1000
const DEFAULT_LBFGS_STARTS = 10
const DEFAULT_PROB_SAMPLES = 5000

function safeget(kwargs::Base.Iterators.Pairs, key::Symbol, default)
    return haskey(kwargs, key) ? kwargs[key] : default
end
function safeget(kwargs::Dict, key::Symbol, default)
    return haskey(kwargs, key) ? kwargs[key] : default
end
function safeget(kwargs::NamedTuple, key::Symbol, default)
    return haskey(kwargs, key) ? getfield(kwargs, key) : default
end


"""
Helper function to find the maximum of func_to_max within bounds.
"""
function find_maximum(func_to_max::Function,
    bounds::AbstractMatrix,
    maximizer::MaximizerType;
    kwargs...
)

    dims = size(bounds, 1)
    best_x = Vector{Float64}(undef, dims)
    max_func_value = -Inf

    grid_points_per_dim = safeget(kwargs, :grid_points_per_dim, DEFAULT_GRID_POINTS)
    random_samples = safeget(kwargs, :random_samples, DEFAULT_RANDOM_SAMPLES)
    lbfgs_starts = safeget(kwargs, :lbfgs_starts, DEFAULT_LBFGS_STARTS)
    prob_samples = safeget(kwargs, :prob_samples, DEFAULT_PROB_SAMPLES)


    if maximizer == GRID
        candidate_points = generate_grid(bounds, grid_points_per_dim)
        if size(candidate_points, 2) == 0
            error("GRID: No candidate points generated.")
        end
        func_values = mapslices(func_to_max, candidate_points, dims=1)[:]
        max_func_value, best_idx = findmax(func_values)
        best_x = candidate_points[:, best_idx]

    elseif maximizer == RANDOM
        candidate_points = sample_randomly(bounds, random_samples)
        if size(candidate_points, 2) == 0
            error("RANDOM: No candidate points generated.")
        end
        func_values = mapslices(func_to_max, candidate_points, dims=1)[:]
        max_func_value, best_idx = findmax(func_values)
        best_x = candidate_points[:, best_idx]

    elseif maximizer == LBFGS
        lower_bounds = bounds[:, 1]
        upper_bounds = bounds[:, 2]
        best_result_value = -Inf
        local best_result_minimizer = zeros(dims)
        neg_func = x -> -func_to_max(x)
        lbfgs_converged_count = 0
        for i in 1:lbfgs_starts
            initial_x = sample_randomly(bounds, 1)[:, 1]
            try
                result = optimize(neg_func, lower_bounds, upper_bounds, initial_x, Fminbox(LBFGS()), Optim.Options(iterations=100, f_tol=1e-6, show_trace=false))
                if Optim.converged(result)
                    lbfgs_converged_count += 1
                    current_val = -Optim.minimum(result)
                    if current_val > best_result_value
                        best_result_value = current_val
                        best_result_minimizer = Optim.minimizer(result)
                    end
                end
            catch e
                continue
            end
        end
        if best_result_value > -Inf
            max_func_value = best_result_value
            best_x = best_result_minimizer
        else
            println("Warning: LBFGS failed, falling back to RANDOM search.")
            candidate_points = sample_randomly(bounds, random_samples)
            func_values = mapslices(func_to_max, candidate_points, dims=1)[:]
            max_func_value, best_idx = findmax(func_values)
            best_x = candidate_points[:, best_idx]
        end

    elseif maximizer == PROBSAMPLE
        candidate_points = sample_randomly(bounds, prob_samples)
        if size(candidate_points, 2) == 0
            error("PROBSAMPLE: No candidates.")
        end
        func_values = mapslices(func_to_max, candidate_points, dims=1)[:]
        min_val = minimum(func_values)
        weights = max.(0.0, func_values .- min_val) .+ 1e-9
        if sum(weights) < 1e-9 || any(!isfinite, weights)
            weights = ones(length(func_values))
        end
        try
            prob_weights = Weights(weights, sum(weights))
            chosen_idx = sample(1:length(func_values), prob_weights)
            best_x = candidate_points[:, chosen_idx]
            max_func_value = func_values[chosen_idx]
        catch e
            println("Error during PROBSAMPLE: $e. Falling back to RANDOM max.")
            max_func_value, best_idx = findmax(func_values)
            best_x = candidate_points[:, best_idx]
        end

    else
        error("Unknown maximizer type: $maximizer")
    end

    best_x = max.(min.(best_x, bounds[:, 2]), bounds[:, 1])
    return best_x, max_func_value
end


"""
    maximize_acquisition(...) - For Bayesian Optimization (GP)
"""
function maximize_acquisition(gp::GPE,
    bounds::AbstractMatrix,
    acq_type::AcquisitionFunctions.AcquisitionType,
    maximizer::MaximizerType,
    maximize::Bool,
    y_best::Real;
    kwargs...)

    dims = size(bounds, 1)
    acq_func_wrapper = x -> begin
        x_mat = reshape(x, dims, 1)
        val = AcquisitionFunctions.calculate_acquisition(gp, x_mat, acq_type, maximize, y_best;
            ξ=safeget(kwargs, :ξ, 0.01),
            κ=safeget(kwargs, :κ, 2.0))
        return val[1]
    end

    maximizer_kwargs = filter(p -> first(p) in (:grid_points_per_dim, :random_samples, :lbfgs_starts, :prob_samples), kwargs)
    best_x, max_acq_value = find_maximum(acq_func_wrapper, bounds, maximizer; maximizer_kwargs...)

    return best_x, max_acq_value
end


"""
    find_max_variance_point(...) - For Active Learning (GP)
"""
function find_max_variance_point(gp::GPE, bounds::AbstractMatrix, maximizer::MaximizerType; kwargs...)
    dims = size(bounds, 1)
    variance_func = x -> begin
        _, var = GaussianProcesses.predict_y(gp, reshape(x, dims, 1))
        return max(var[1], 0.0)
    end
    maximizer_kwargs = filter(p -> first(p) in (:grid_points_per_dim, :random_samples, :lbfgs_starts), kwargs)
    best_x, max_var_value = find_maximum(variance_func, bounds, maximizer; maximizer_kwargs...)
    return best_x, max_var_value
end


"""
    find_max_rf_variance_point(...) - For Active Learning (Random Forest)
"""
function find_max_rf_variance_point(model::RandomForestModel.RandomForestRegressor,
    bounds::AbstractMatrix,
    maximizer::MaximizerType;
    kwargs...)

    dims = size(bounds, 1)
    rf_variance_func = x -> begin
        _, variances = RandomForestModel.predict_rf_mean_variance(model, reshape(x, dims, 1))
        return max(variances[1], 0.0)
    end
    maximizer_kwargs = filter(p -> first(p) in (:grid_points_per_dim, :random_samples, :lbfgs_starts), kwargs)
    if maximizer == PROBSAMPLE
        println("Warning: PROBSAMPLE maximizer might not be ideal for RF variance maximization.")
        prob_samples = safeget(kwargs, :prob_samples, DEFAULT_PROB_SAMPLES)
        maximizer_kwargs = (; maximizer_kwargs..., prob_samples=prob_samples)
    end
    best_x, max_rf_var_value = find_maximum(rf_variance_func, bounds, maximizer; maximizer_kwargs...)
    return best_x, max_rf_var_value
end


end