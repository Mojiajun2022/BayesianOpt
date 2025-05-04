module OptimizationLoop

using LinearAlgebra
using Random
using Printf
using Statistics
using GaussianProcesses
using DecisionTree
using ..Utils
using ..GPModel
using ..RandomForestModel
using ..AcquisitionFunctions
using ..Maximizers
using Distributions

export bayes_optimize, active_learn, active_learn_rf

"""
    bayes_optimize(objective_func, bounds, n_iterations; kwargs...)

Performs Bayesian Optimization to find the optimum of the objective function.

Args:
    objective_func (Function): The function to optimize. Takes a vector input, returns a scalar.
    bounds (AbstractMatrix): d x 2 matrix of lower and upper bounds for each dimension.
    n_iterations (Int): Number of optimization iterations (after initial design).

Keyword Args:
    n_initial_points (Int, N): Number of points for the initial design.
    initial_design (Symbol, :random): Strategy for initial design (:random, :grid).
    X_init (Matrix, nothing): Pre-defined initial X points.
    Y_init (Vector, nothing): Pre-defined initial Y values (must match X_init).
    kernel: Initial GP kernel (e.g., from GaussianProcesses.jl). Set priors directly on this object if desired.
    logNoise (Real, -2.0): Initial log observation noise variance.
    mean_func: GP mean function.
    optimize_gp_hypers (Bool, true): Whether to optimize GP hyperparameters.
    logNoise_prior (Union{Distribution, Nothing}, nothing): Prior distribution for logNoise (from Distributions.jl).
    acquisition_func (AcquisitionType, EI): Acquisition function type (EI, PI, UCB).
    acq_maximizer (MaximizerType, LBFGS): Strategy to maximize acquisition function (GRID, RANDOM, LBFGS, PROBSAMPLE).
    acq_ξ (Real, 0.01): ξ parameter for EI/PI acquisition functions.
    acq_κ (Real, 2.0): κ parameter for UCB acquisition function.
    maximizer_grid_points_per_dim (Int): Points per dimension if acq_maximizer=GRID.
    maximizer_random_samples (Int): Samples if acq_maximizer=RANDOM.
    maximizer_lbfgs_starts (Int): Random starts if acq_maximizer=LBFGS.
    maximizer_prob_samples (Int): Samples if acq_maximizer=PROBSAMPLE.
    maximize (Bool, false): Set to true to maximize objective_func, false to minimize.
    verbose (Bool, true): Print progress information.
    random_seed (Int, nothing): Seed for reproducibility.

Returns:
    A NamedTuple containing optimization results (best_x, best_y, X_history, Y_history, final_gp, etc.).
"""
function bayes_optimize(objective_func::Function,
    bounds::AbstractMatrix,
    n_iterations::Int;
    n_initial_points::Int=5,
    initial_design::Symbol=:random,
    X_init=nothing, Y_init=nothing,
    kernel=GaussianProcesses.SEIso(0.0, 0.0),
    logNoise::Real=-2.0,
    mean_func=GaussianProcesses.MeanZero(),
    optimize_gp_hypers::Bool=true,
    logNoise_prior::Union{Distribution,Nothing}=nothing,
    acquisition_func::AcquisitionFunctions.AcquisitionType=EI, # Qualified type
    acq_maximizer::Maximizers.MaximizerType=LBFGS, # Qualified type
    acq_ξ::Real=0.01,
    acq_κ::Real=2.0,
    maximizer_grid_points_per_dim::Int=Maximizers.DEFAULT_GRID_POINTS,
    maximizer_random_samples::Int=Maximizers.DEFAULT_RANDOM_SAMPLES,
    maximizer_lbfgs_starts::Int=Maximizers.DEFAULT_LBFGS_STARTS,
    maximizer_prob_samples::Int=Maximizers.DEFAULT_PROB_SAMPLES,
    maximize::Bool=false,
    verbose::Bool=true,
    random_seed=nothing)
    if random_seed !== nothing
        Random.seed!(random_seed)
    end
    dims = size(bounds, 1)
    if dims == 0
        error("Bounds must have at least one dimension.")
    end

    local X_history_mat::Matrix{Float64}
    local Y_history_vec::Vector{Float64}
    local actual_n_initial_points::Int

    if X_init === nothing || Y_init === nothing
        actual_n_initial_points = n_initial_points
        if verbose
            println("Generating $actual_n_initial_points initial design points ($initial_design)...")
        end
        if initial_design == :random
            X_history_mat = Utils.sample_randomly(bounds, actual_n_initial_points)
        elseif initial_design == :grid
            points_per_dim_init = max(2, round(Int, actual_n_initial_points^(1 / dims)))
            X_history_mat = Utils.generate_grid(bounds, points_per_dim_init)
            actual_n_initial_points = size(X_history_mat, 2)
            if verbose
                println("  Adjusted initial points to $actual_n_initial_points based on grid dimensions.")
            end
        else
            error("Unknown initial design: $initial_design")
        end

        if size(X_history_mat, 2) == 0
            error("Failed to generate initial points.")
        end
        Y_history_vec = Vector{Float64}(undef, actual_n_initial_points)
        for i in 1:actual_n_initial_points
            Y_history_vec[i] = objective_func(X_history_mat[:, i])
            if verbose
                @printf("  Initial point %d/%d: f(%s) = %.4e\n", i, actual_n_initial_points, string(round.(X_history_mat[:, i], digits=3)), Y_history_vec[i])
            end
        end
    else
        if verbose
            println("Using provided initial data.")
        end
        if size(X_init, 1) != dims
            error("Provided X_init dim mismatch.")
        end
        if size(X_init, 2) != length(Y_init)
            error("Provided X_init/Y_init size mismatch.")
        end
        X_history_mat = copy(X_init)
        Y_history_vec = Vector{Float64}(copy(Y_init))
        actual_n_initial_points = size(X_history_mat, 2)
    end

    gp = nothing
    current_kernel = kernel
    current_logNoise = logNoise

    if verbose
        println("\nStarting Bayesian Optimization loop ($n_iterations iterations)...")
    end
    for iter = 1:n_iterations
        if verbose
            @printf("\n--- BO Iteration %d/%d ---\n", iter, n_iterations)
        end
        if verbose
            println("Updating GP model...")
        end


        gp = GPModel.update_gp!(gp, X_history_mat, Y_history_vec;
            kernel=current_kernel,
            logNoise=current_logNoise,
            mean_func=mean_func,
            optimize_hypers=optimize_gp_hypers,
            logNoise_prior=logNoise_prior
        )

        if gp === nothing
            error("Failed to update GP model in iteration $iter.")
        end
        current_kernel = gp.kernel
        current_logNoise = gp.logNoise

        y_best = Utils.get_incumbent(Y_history_vec, maximize)
        if verbose
            @printf("Current best observed y_best = %.4e\n", y_best)
        end

        if verbose
            println("Maximizing acquisition function ($acquisition_func) using ($acq_maximizer)...")
        end
        x_next, acq_val = Maximizers.maximize_acquisition(
            gp, bounds, acquisition_func, acq_maximizer, maximize, y_best;
            ξ=acq_ξ, κ=acq_κ,
            grid_points_per_dim=maximizer_grid_points_per_dim,
            random_samples=maximizer_random_samples,
            lbfgs_starts=maximizer_lbfgs_starts,
            prob_samples=maximizer_prob_samples
        )
        if verbose
            @printf("Next point suggested x_next = %s (acq_val = %.4e)\n", string(round.(x_next, digits=4)), acq_val)
        end
        if verbose
            println("Evaluating objective function f(x_next)...")
        end
        y_next = objective_func(x_next)
        if verbose
            @printf("Evaluation result y_next = %.4e\n", y_next)
        end
        X_history_mat = hcat(X_history_mat, x_next)
        push!(Y_history_vec, y_next)
    end

    if verbose
        println("\nBayesian Optimization finished.")
    end
    final_y_best, best_idx = maximize ? findmax(Y_history_vec) : findmin(Y_history_vec)
    final_x_best = X_history_mat[:, best_idx]
    if verbose
        println("="^30)
        println("Optimization Results:")
        println("  Total Iterations: $n_iterations (+ $actual_n_initial_points initial)")
        println("  Best Found x: ", string(round.(final_x_best, digits=4)))
        println("  Best Found y: ", final_y_best)
        println("="^30)
    end

    return (best_x=final_x_best, best_y=final_y_best, X_history=X_history_mat, Y_history=Y_history_vec,
        final_gp=gp, bounds=bounds, maximize=maximize, acquisition_func=acquisition_func,
        n_iterations=n_iterations, n_initial_points=actual_n_initial_points,
        model_type=:GP
    )
end


"""
    active_learn(query_func, bounds, n_iterations; kwargs...)

Performs Active Learning using a Gaussian Process model to efficiently learn the query function.

Args:
    query_func (Function): The function to query and model.
    bounds (AbstractMatrix): d x 2 matrix of bounds.
    n_iterations (Int): Number of active learning iterations.

Keyword Args:
    n_initial_points (Int, N): Number of initial random points.
    initial_design (Symbol, :random): Strategy for initial design.
    X_init (Matrix, nothing): Pre-defined initial X points.
    Y_init (Vector, nothing): Pre-defined initial Y values.
    X_test (Matrix, nothing): Test set X data for evaluating model performance.
    Y_test (Vector, nothing): Test set Y data.
    kernel: Initial GP kernel. Set priors directly on this object if desired.
    logNoise (Real, -2.0): Initial log observation noise variance.
    mean_func: GP mean function.
    optimize_gp_hypers (Bool, true): Whether to optimize GP hyperparameters.
    logNoise_prior (Union{Distribution, Nothing}, nothing): Prior distribution for logNoise.
    variance_maximizer (MaximizerType, LBFGS): Strategy to find point of maximum variance.
    maximizer_grid_points_per_dim (Int): Points per dimension if variance_maximizer=GRID.
    maximizer_random_samples (Int): Samples if variance_maximizer=RANDOM.
    maximizer_lbfgs_starts (Int): Random starts if variance_maximizer=LBFGS.
    verbose (Bool, true): Print progress information.
    random_seed (Int, nothing): Seed for reproducibility.

Returns:
    A NamedTuple containing learning results (X_history, Y_history, final_gp, test_rmse_history, etc.).
"""
function active_learn(query_func::Function,
    bounds::AbstractMatrix,
    n_iterations::Int;
    n_initial_points::Int=5,
    initial_design::Symbol=:random,
    X_init=nothing, Y_init=nothing,
    X_test=nothing, Y_test=nothing,
    kernel=GaussianProcesses.SEIso(0.0, 0.0),
    logNoise::Real=-2.0,
    mean_func=GaussianProcesses.MeanZero(),
    optimize_gp_hypers::Bool=true,
    logNoise_prior::Union{Distribution,Nothing}=nothing,
    variance_maximizer::Maximizers.MaximizerType=LBFGS,
    maximizer_grid_points_per_dim::Int=Maximizers.DEFAULT_GRID_POINTS,
    maximizer_random_samples::Int=Maximizers.DEFAULT_RANDOM_SAMPLES,
    maximizer_lbfgs_starts::Int=Maximizers.DEFAULT_LBFGS_STARTS,
    verbose::Bool=true,
    random_seed=nothing)

    if random_seed !== nothing
        Random.seed!(random_seed)
    end
    dims = size(bounds, 1)
    if dims == 0
        error("Bounds must have at least one dimension.")
    end
    run_test_eval = false
    test_rmse_history = Float64[]
    if X_test !== nothing && Y_test !== nothing
        if size(X_test, 1) != dims
            error("X_test dim mismatch.")
        end
        if size(X_test, 2) != length(Y_test)
            error("X_test/Y_test size mismatch.")
        end
        run_test_eval = true
        if verbose
            println("Test set provided, performance will be tracked.")
        end
    elseif X_test !== nothing || Y_test !== nothing
        println("Warning: Both X_test and Y_test must be provided to track test performance.")
    end

    local X_train_mat::Matrix{Float64}
    local Y_train_vec::Vector{Float64}
    local actual_n_initial_points::Int

    if X_init === nothing || Y_init === nothing
        actual_n_initial_points = n_initial_points
        if verbose
            println("Generating $actual_n_initial_points initial training points ($initial_design)...")
        end
        if initial_design == :random
            X_train_mat = Utils.sample_randomly(bounds, actual_n_initial_points)
        elseif initial_design == :grid
            points_per_dim_init = max(2, round(Int, actual_n_initial_points^(1 / dims)))
            X_train_mat = Utils.generate_grid(bounds, points_per_dim_init)
            actual_n_initial_points = size(X_train_mat, 2)
            if verbose
                println("  Adjusted initial points to $actual_n_initial_points based on grid dimensions.")
            end
        else
            error("Unknown initial design: $initial_design")
        end

        if size(X_train_mat, 2) == 0
            error("Failed to generate initial points.")
        end
        Y_train_vec = Vector{Float64}(undef, actual_n_initial_points)
        for i in 1:actual_n_initial_points
            Y_train_vec[i] = query_func(X_train_mat[:, i])
            if verbose
                @printf("  Initial point %d: f(%s)=%.4e\n", i, string(round.(X_train_mat[:, i], digits=3)), Y_train_vec[i])
            end
        end
    else
        if verbose
            println("Using provided initial data for training.")
        end
        if size(X_init, 1) != dims
            error("X_init dim mismatch.")
        end
        if size(X_init, 2) != length(Y_init)
            error("X_init/Y_init size mismatch.")
        end
        X_train_mat = copy(X_init)
        Y_train_vec = Vector{Float64}(copy(Y_init))
        actual_n_initial_points = size(X_train_mat, 2)
    end

    gp = nothing
    current_kernel = kernel
    current_logNoise = logNoise

    if verbose
        println("\nStarting Active Learning loop ($n_iterations iterations)...")
    end
    for iter = 1:n_iterations
        if verbose
            @printf("\n--- AL Iteration %d/%d ---\n", iter, n_iterations)
        end
        if verbose
            println("Updating GP model on training data...")
        end

        gp = GPModel.update_gp!(gp, X_train_mat, Y_train_vec;
            kernel=current_kernel,
            logNoise=current_logNoise,
            mean_func=mean_func,
            optimize_hypers=optimize_gp_hypers,
            logNoise_prior=logNoise_prior
        )

        if gp === nothing
            error("Failed to update GP model in iteration $iter.")
        end
        current_kernel = gp.kernel
        current_logNoise = gp.logNoise

        if run_test_eval
            if verbose
                print("Evaluating GP on test set... ")
            end
            try
                mu_test, _ = GPModel.predict_with_gp(gp, X_test)
                test_rmse = sqrt(mean((mu_test .- Y_test) .^ 2))
                push!(test_rmse_history, test_rmse)
                if verbose
                    @printf("Test RMSE = %.4e\n", test_rmse)
                end
            catch e
                println("\nError evaluating GP on test set in iteration $iter: $e.")
                push!(test_rmse_history, NaN)
            end
        end

        if verbose
            println("Finding point with maximum predictive variance using ($variance_maximizer)...")
        end
        x_next, max_var_val = Maximizers.find_max_variance_point(
            gp, bounds, variance_maximizer;
            grid_points_per_dim=maximizer_grid_points_per_dim,
            random_samples=maximizer_random_samples,
            lbfgs_starts=maximizer_lbfgs_starts
        )
        if verbose
            @printf("Next point suggested x_next = %s (max_variance = %.4e)\n", string(round.(x_next, digits=4)), max_var_val)
        end
        if verbose
            println("Querying function f(x_next)...")
        end
        y_next = query_func(x_next)
        if verbose
            @printf("Query result y_next = %.4e\n", y_next)
        end
        is_duplicate = any(norm(X_train_mat[:, i] - x_next) < 1e-7 for i in 1:size(X_train_mat, 2))
        if is_duplicate
            println("Warning: Skipping duplicate point suggestion in iter $iter: $x_next")
            continue
        end
        X_train_mat = hcat(X_train_mat, x_next)
        push!(Y_train_vec, y_next)
    end

    if verbose
        println("\nActive Learning finished.")
    end
    if verbose
        println("="^30)
        println("Active Learning Results:")
        println("  Total Iterations: $n_iterations (+ $actual_n_initial_points initial)")
        println("  Final training set size: ", size(X_train_mat, 2))
        if run_test_eval && !isempty(test_rmse_history)
            println("  Final Test RMSE: ", round(test_rmse_history[end], digits=4))
        end
        println("="^30)
    end

    results_dict = Dict{Symbol,Any}(
        :X_history => X_train_mat,
        :Y_history => Y_train_vec,
        :final_gp => gp,
        :bounds => bounds,
        :n_iterations => n_iterations,
        :n_initial_points => actual_n_initial_points
    )
    if run_test_eval
        results_dict[:test_rmse_history] = test_rmse_history
        results_dict[:X_test] = X_test
        results_dict[:Y_test] = Y_test
    end
    results_dict[:model_type] = :GP
    return NamedTuple(results_dict)
end

"""
    active_learn_rf(...) - Active Learning with Random Forest
"""
function active_learn_rf(query_func::Function,
    bounds::AbstractMatrix,
    n_iterations::Int;
    n_initial_points::Int=5, initial_design::Symbol=:random, X_init=nothing, Y_init=nothing,
    X_test=nothing, Y_test=nothing,
    rf_n_trees::Int=100, rf_kwargs=Dict(),
    variance_maximizer::Maximizers.MaximizerType=LBFGS,
    maximizer_kwargs=Dict(),
    verbose::Bool=true, random_seed=nothing)

    if random_seed !== nothing
        Random.seed!(random_seed)
    end
    dims = size(bounds, 1)
    if dims == 0
        error("Bounds invalid.")
    end
    run_test_eval = false
    test_rmse_history = Float64[]
    if X_test !== nothing && Y_test !== nothing
        if size(X_test, 1) != dims
            error("X_test dim mismatch.")
        end
        if size(X_test, 2) != length(Y_test)
            error("X_test/Y_test size mismatch.")
        end
        run_test_eval = true
        if verbose
            println("Test set provided...")
        end
    elseif X_test !== nothing || Y_test !== nothing
        println("Warning: Both X_test/Y_test needed...")
    end

    local X_train_mat::Matrix{Float64}
    local Y_train_vec::Vector{Float64}
    local actual_n_initial_points::Int

    if X_init === nothing || Y_init === nothing
        actual_n_initial_points = n_initial_points
        if verbose
            println("Generating $actual_n_initial_points initial training points ($initial_design)...")
        end
        if initial_design == :random
            X_train_mat = Utils.sample_randomly(bounds, actual_n_initial_points)
        elseif initial_design == :grid
            points_per_dim_init = max(2, round(Int, actual_n_initial_points^(1 / dims)))
            X_train_mat = Utils.generate_grid(bounds, points_per_dim_init)
            actual_n_initial_points = size(X_train_mat, 2)
            if verbose
                println("  Adjusted to $actual_n_initial_points grid points.")
            end
        else
            error("Unknown initial design: $initial_design")
        end
        if size(X_train_mat, 2) == 0
            error("Failed to generate initial points.")
        end
        Y_train_vec = [query_func(X_train_mat[:, i]) for i in 1:actual_n_initial_points]
        if verbose
            for i = 1:actual_n_initial_points
                @printf("  Initial point %d: f(%s)=%.4e\n", i, string(round.(X_train_mat[:, i], digits=3)), Y_train_vec[i])
            end
        end
    else
        if verbose
            println("Using provided initial data for training.")
        end
        if size(X_init, 1) != dims
            error("X_init dim mismatch.")
        end
        if size(X_init, 2) != length(Y_init)
            error("X_init/Y_init size mismatch.")
        end
        X_train_mat = copy(X_init)
        Y_train_vec = Vector{Float64}(copy(Y_init))
        actual_n_initial_points = size(X_train_mat, 2)
    end

    rf_model = nothing

    if verbose
        println("\nStarting RF Active Learning loop ($n_iterations iterations)...")
    end
    for iter = 1:n_iterations
        if verbose
            @printf("\n--- RF AL Iteration %d/%d ---\n", iter, n_iterations)
        end
        if verbose
            println("Training Random Forest model...")
        end
        rf_model = RandomForestModel.train_rf(X_train_mat, Y_train_vec; n_trees=rf_n_trees, rf_kwargs...)
        if rf_model === nothing
            error("Failed to train RF model in iteration $iter.")
        end
        n_trees_trained = try
            rf_model.n_trees
        catch
            rf_n_trees
        end
        if verbose
            println("RF model trained with $(n_trees_trained) trees.")
        end
        if run_test_eval
            if verbose
                print("Evaluating RF on test set... ")
            end
            try
                mu_test, _ = RandomForestModel.predict_rf_mean_variance(rf_model, X_test)
                test_rmse = sqrt(mean((mu_test .- Y_test) .^ 2))
                push!(test_rmse_history, test_rmse)
                if verbose
                    @printf("Test RMSE = %.4e\n", test_rmse)
                end
            catch e
                println("\nError evaluating RF on test set in iteration $iter: $e.")
                push!(test_rmse_history, NaN)
            end
        end

        if verbose
            println("Finding point with maximum RF prediction variance using ($variance_maximizer)...")
        end
        merged_maximizer_kwargs = Dict{Symbol,Any}()
        merged_maximizer_kwargs[:grid_points_per_dim] = Maximizers.DEFAULT_GRID_POINTS
        merged_maximizer_kwargs[:random_samples] = Maximizers.DEFAULT_RANDOM_SAMPLES
        merged_maximizer_kwargs[:lbfgs_starts] = Maximizers.DEFAULT_LBFGS_STARTS
        merged_maximizer_kwargs[:prob_samples] = Maximizers.DEFAULT_PROB_SAMPLES
        merge!(merged_maximizer_kwargs, Dict(maximizer_kwargs))
        x_next, max_rf_var = Maximizers.find_max_rf_variance_point(rf_model, bounds, variance_maximizer; merged_maximizer_kwargs...)
        if verbose
            @printf("Next point suggested x_next = %s (max_rf_variance = %.4e)\n", string(round.(x_next, digits=4)), max_rf_var)
        end

        if verbose
            println("Querying function f(x_next)...")
        end
        y_next = query_func(x_next)
        if verbose
            @printf("Query result y_next = %.4e\n", y_next)
        end

        is_duplicate = any(norm(X_train_mat[:, i] - x_next) < 1e-7 for i in 1:size(X_train_mat, 2))
        if is_duplicate
            println("Warning: Skipping duplicate point suggestion in iter $iter: $x_next")
            continue
        end
        X_train_mat = hcat(X_train_mat, x_next)
        push!(Y_train_vec, y_next)
    end

    if verbose
        println("\nRF Active Learning finished.")
    end
    if verbose
        println("="^30)
        println("RF Active Learning Results:")
        println("  Total Iterations: $n_iterations (+ $actual_n_initial_points initial)")
        println("  Final training set size: ", size(X_train_mat, 2))
        if run_test_eval && !isempty(test_rmse_history)
            println("  Final Test RMSE: ", round(test_rmse_history[end], digits=4))
        end
        println("="^30)
    end

    results_dict = Dict{Symbol,Any}(
        :X_history => X_train_mat,
        :Y_history => Y_train_vec,
        :final_model => rf_model,
        :model_type => :RandomForest,
        :bounds => bounds,
        :n_iterations => n_iterations,
        :n_initial_points => actual_n_initial_points)
    if run_test_eval
        results_dict[:test_rmse_history] = test_rmse_history
        results_dict[:X_test] = X_test
        results_dict[:Y_test] = Y_test
    end

    return NamedTuple(results_dict)

end


function active_learn_residual_rf_gp(query_func::Function,
    bounds::AbstractMatrix,
    n_iterations::Int;
    n_initial_points::Int=10,
    initial_design::Symbol=:random,
    X_init=nothing, Y_init=nothing,
    X_test=nothing, Y_test=nothing,
    rf_n_trees::Int=100,
    rf_kwargs=Dict(),
    kernel=GaussianProcesses.SEIso(0.0, 0.0),
    logNoise::Real=-2.0,
    optimize_gp_hypers::Bool=true,
    logNoise_prior::Union{Distribution,Nothing}=nothing,
    variance_maximizer::Maximizers.MaximizerType=LBFGS,
    maximizer_kwargs=Dict(),
    verbose::Bool=true,
    random_seed=nothing)

    if random_seed !== nothing
        Random.seed!(random_seed)
    end
    dims = size(bounds, 1)
    if dims == 0
        error("Bounds must have at least one dimension.")
    end
    run_test_eval = false
    test_rmse_history = Float64[]
    if X_test !== nothing && Y_test !== nothing
        if size(X_test, 1) != dims
            error("X_test dim mismatch.")
        end
        if size(X_test, 2) != length(Y_test)
            error("X_test/Y_test size mismatch.")
        end
        run_test_eval = true
        if verbose
            println("Test set provided, performance will be tracked.")
        end
    elseif X_test !== nothing || Y_test !== nothing
        println("Warning: Both X_test and Y_test must be provided to track test performance.")
    end

    local X_history_mat::Matrix{Float64}
    local Y_history_vec::Vector{Float64}
    local actual_n_initial_points::Int

    if X_init === nothing || Y_init === nothing
        actual_n_initial_points = n_initial_points
        if verbose
            println("Generating $actual_n_initial_points initial training points ($initial_design)...")
        end
        if initial_design == :random
            X_history_mat = Utils.sample_randomly(bounds, actual_n_initial_points)
        elseif initial_design == :grid # Only practical for low-dim
            points_per_dim_init = max(2, round(Int, actual_n_initial_points^(1 / dims)))
            X_history_mat = Utils.generate_grid(bounds, points_per_dim_init)
            actual_n_initial_points = size(X_history_mat, 2)
            if verbose
                println("  Adjusted initial points to $actual_n_initial_points based on grid dimensions.")
            end
        else
            error("Unknown initial design: $initial_design")
        end

        if size(X_history_mat, 2) == 0
            error("Failed to generate initial points.")
        end
        Y_history_vec = Vector{Float64}(undef, actual_n_initial_points)
        for i in 1:actual_n_initial_points
            Y_history_vec[i] = query_func(X_history_mat[:, i])
            if verbose
                @printf("  Initial point %d: f(%s)=%.4e\n", i, string(round.(X_history_mat[:, i], digits=3)), Y_history_vec[i])
            end
        end
    else
        if verbose
            println("Using provided initial data for training.")
        end
        if size(X_init, 1) != dims
            error("X_init dim mismatch.")
        end
        if size(X_init, 2) != length(Y_init)
            error("X_init/Y_init size mismatch.")
        end
        X_history_mat = copy(X_init)
        Y_history_vec = Vector{Float64}(copy(Y_init))
        actual_n_initial_points = size(X_history_mat, 2)
    end

    rf_model = nothing
    residual_gp = nothing
    current_kernel = kernel
    current_logNoise = logNoise
    if verbose
        println("\nStarting Residual RF+GP Active Learning loop ($n_iterations iterations)...")
    end
    for iter = 1:n_iterations
        if verbose
            @printf("\n--- Residual RF+GP AL Iteration %d/%d ---\n", iter, n_iterations)
        end
        if verbose
            println("Training Random Forest model...")
        end
        rf_model = RandomForestModel.train_rf(X_history_mat, Y_history_vec; n_trees=rf_n_trees, rf_kwargs...)
        if rf_model === nothing
            error("Failed to train RF model in iteration $iter.")
        end
        n_trees_trained = try
            rf_model.n_trees
        catch
            rf_n_trees
        end
        if verbose
            println("RF model trained with $(n_trees_trained) trees.")
        end

        if verbose
            println("Calculating RF predictions on training data...")
        end
        rf_preds_train, _ = RandomForestModel.predict_rf_mean_variance(rf_model, X_history_mat)
        Y_residuals = Y_history_vec .- rf_preds_train
        if verbose
            println("Calculated residuals for GP training.")
        end

        if verbose
            println("Updating Residual GP model...")
        end
        residual_gp = GPModel.update_gp!(residual_gp, X_history_mat, Y_residuals; # Fit GP on residuals
            kernel=current_kernel,
            logNoise=current_logNoise,
            mean_func=MeanZero(),
            optimize_hypers=optimize_gp_hypers,
            logNoise_prior=logNoise_prior
        )
        if residual_gp === nothing
            error("Failed to update Residual GP model in iteration $iter.")
        end
        current_kernel = residual_gp.kernel
        current_logNoise = residual_gp.logNoise

        if run_test_eval
            if verbose
                print("Evaluating Combined RF+GP Model on test set... ")
            end
            try
                rf_preds_test, _ = RandomForestModel.predict_rf_mean_variance(rf_model, X_test)
                gp_preds_test, _ = GPModel.predict_with_gp(residual_gp, X_test)
                final_preds_test = rf_preds_test .+ gp_preds_test
                test_rmse = sqrt(mean((final_preds_test .- Y_test) .^ 2))
                push!(test_rmse_history, test_rmse)
                if verbose
                    @printf("Test RMSE = %.4e\n", test_rmse)
                end
            catch e
                println("\nError evaluating combined model on test set in iteration $iter: $e.")
                push!(test_rmse_history, NaN)
            end
        end

        if verbose
            println("Finding point with maximum residual GP variance using ($variance_maximizer)...")
        end
        maximizer_options = Dict{Symbol,Any}()
        if haskey(maximizer_kwargs, :grid_points_per_dim)
            maximizer_options[:grid_points_per_dim] = maximizer_kwargs[:grid_points_per_dim]
        end
        if haskey(maximizer_kwargs, :random_samples)
            maximizer_options[:random_samples] = maximizer_kwargs[:random_samples]
        end
        if haskey(maximizer_kwargs, :lbfgs_starts)
            maximizer_options[:lbfgs_starts] = maximizer_kwargs[:lbfgs_starts]
        end
        if haskey(maximizer_kwargs, :prob_samples)
            maximizer_options[:prob_samples] = maximizer_kwargs[:prob_samples]
        end

        x_next, max_gp_var_val = Maximizers.find_max_variance_point(
            residual_gp, bounds, variance_maximizer;
            maximizer_options...
        )
        if verbose
            @printf("Next point suggested x_next = %s (max_gp_variance = %.4e)\n", string(round.(x_next, digits=4)), max_gp_var_val)
        end

        if verbose
            println("Querying function f(x_next)...")
        end
        y_next = query_func(x_next)
        if verbose
            @printf("Query result y_next = %.4e\n", y_next)
        end
        is_duplicate = any(norm(X_history_mat[:, i] - x_next) < 1e-7 for i in 1:size(X_history_mat, 2))
        if is_duplicate
            println("Warning: Skipping duplicate point suggestion in iter $iter: $x_next")
            continue
        end
        X_history_mat = hcat(X_history_mat, x_next)
        push!(Y_history_vec, y_next)

    end

    if verbose
        println("\nResidual RF+GP Active Learning finished.")
    end
    if verbose
        println("="^30)
        println("Residual RF+GP Active Learning Results:")
        println("  Total Iterations: $n_iterations (+ $actual_n_initial_points initial)")
        println("  Final training set size: ", size(X_history_mat, 2))
        if run_test_eval && !isempty(test_rmse_history)
            println("  Final Test RMSE: ", round(filter(!isnan, test_rmse_history)[end], digits=4))
        end
        println("="^30)
    end

    results_dict = Dict{Symbol,Any}(
        :X_history => X_history_mat,
        :Y_history => Y_history_vec,
        :final_rf_model => rf_model,
        :final_residual_gp => residual_gp,
        :model_type => :ResidualRFGP,
        :bounds => bounds,
        :n_iterations => n_iterations,
        :n_initial_points => actual_n_initial_points
    )
    if run_test_eval
        results_dict[:test_rmse_history] = test_rmse_history
        results_dict[:X_test] = X_test
        results_dict[:Y_test] = Y_test
    end

    return NamedTuple(results_dict)

end

end

