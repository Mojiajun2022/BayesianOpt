module Visualization

using Plots
using Printf
using LinearAlgebra
using Statistics
using GaussianProcesses
using ..Utils: generate_grid, get_incumbent, sample_randomly
using ..AcquisitionFunctions: calculate_acquisition, AcquisitionType, EI, PI, UCB
using ..GPModel: GPE, predict_with_gp
using ..RandomForestModel

export plot_results

function _safeget(nt::Union{NamedTuple,Dict}, key::Symbol, default=nothing)
    if isa(nt, NamedTuple)
        return haskey(nt, key) ? getfield(nt, key) : default
    elseif isa(nt, Dict)
        return haskey(nt, key) ? nt[key] : default
    else
        return default
    end
end

function _plot_1d_gp(X_history, Y_history, final_gp, bounds, objective_func, best_x, best_y)
    p = plot(title="GP Model vs Observations (1D)", xlabel="x", ylabel="y", framestyle=:box, legend=:outertopright)
    x_range = range(bounds[1, 1], stop=bounds[1, 2], length=200)
    x_plot_mat = reshape(collect(x_range), 1, length(x_range))

    if objective_func !== nothing
        try
            y_true = objective_func.(x_range)
            plot!(p, x_range, y_true, label="True Objective", color=:black, linestyle=:dash, lw=2)
        catch e
            println("Warn: Could not plot 1D objective_func: $e")
        end
    end
    μ_plot, σ_plot = predict_with_gp(final_gp, x_plot_mat)
    plot!(p, x_range, μ_plot, ribbon=1.96 .* σ_plot, label="GP Mean ± 1.96σ", color=:blue, lw=2, fillalpha=0.2)
    scatter!(p, X_history[1, :], Y_history, label="Observations", color=:red, markersize=4, markerstrokewidth=0)
    if best_x !== nothing && best_y !== nothing
        scatter!(p, [best_x[1]], [best_y], label="Best Found", color=:green, markersize=8, marker=:star5)
    end
    return p
end

function _plot_1d_rf(X_history, Y_history, final_model, bounds, objective_func)
    p = plot(title="Random Forest Model vs Observations (1D)", xlabel="x", ylabel="y", framestyle=:box, legend=:outertopright)
    x_range = range(bounds[1, 1], stop=bounds[1, 2], length=200)
    x_plot_mat = reshape(collect(x_range), 1, length(x_range))
    if objective_func !== nothing
        try
            y_true = objective_func.(x_range)
            plot!(p, x_range, y_true, label="True Objective", color=:black, linestyle=:dash, lw=2)
        catch e
            println("Warn: Could not plot 1D objective_func: $e")
        end
    end
    μ_rf_plot, var_rf_plot = RandomForestModel.predict_rf_mean_variance(final_model, x_plot_mat)
    var_rf_plot = max.(var_rf_plot, 0.0)
    σ_rf_plot = sqrt.(var_rf_plot)
    plot!(p, x_range, μ_rf_plot, color=:blue, lw=2, label="RF Mean Prediction")
    plot!(p, x_range, μ_rf_plot, ribbon=1.0 .* σ_rf_plot, label="±1σ (Tree Spread)", color=:blue, fillalpha=0.2, lw=0)
    scatter!(p, X_history[1, :], Y_history, label="Observations", color=:red, markersize=4, markerstrokewidth=0)
    return p
end

function _plot_1d_residual_rf_gp(X_history, Y_history, final_rf, final_gp_res, bounds, objective_func)
    if final_rf === nothing || !isa(final_rf, RandomForestModel.RandomForestRegressor) ||
       final_gp_res === nothing || !isa(final_gp_res, GPE)
        println("Warning: Invalid models provided for 1D ResidualRFGP plot.")
        return nothing
    end

    p = plot(title="Hybrid RF+GP Model vs Observations (1D)", xlabel="x", ylabel="y", framestyle=:box, legend=:outertopright)
    x_range = range(bounds[1, 1], stop=bounds[1, 2], length=200)
    x_plot_mat = reshape(collect(x_range), 1, length(x_range))

    if objective_func !== nothing
        try
            y_true = objective_func.(x_range)
            plot!(p, x_range, y_true, label="True Objective", color=:black, linestyle=:dash, lw=2)
        catch e
            println("Warn: Could not plot 1D objective_func: $e")
        end
    end

    try
        μ_rf_plot, _ = RandomForestModel.predict_rf_mean_variance(final_rf, x_plot_mat)
        μ_gp_res_plot, σ_gp_res_plot = predict_with_gp(final_gp_res, x_plot_mat)
        μ_combined_plot = μ_rf_plot .+ μ_gp_res_plot
        plot!(p, x_range, μ_combined_plot, color=:purple, lw=2, label="Combined RF+GP Mean")
        plot!(p, x_range, μ_combined_plot, ribbon=1.96 .* σ_gp_res_plot,
            label="± 1.96σ (Residual GP)", color=:purple, fillalpha=0.2, lw=0)

    catch e
        println("Error during prediction for 1D ResidualRFGP plot: $e")
    end

    scatter!(p, X_history[1, :], Y_history, label="Observations", color=:red, markersize=4, markerstrokewidth=0)

    return p
end

function _plot_2d_gp(X_history, Y_history, final_gp, bounds, objective_func, best_x, contour_points_per_dim)
    plot_list = []
    x1_range = range(bounds[1, 1], stop=bounds[1, 2], length=contour_points_per_dim)
    x2_range = range(bounds[2, 1], stop=bounds[2, 2], length=contour_points_per_dim)
    X_plot_mat = generate_grid(bounds, contour_points_per_dim)

    μ_plot, σ_plot = predict_with_gp(final_gp, X_plot_mat)
    μ_grid = reshape(μ_plot, contour_points_per_dim, contour_points_per_dim)
    σ_grid = reshape(σ_plot, contour_points_per_dim, contour_points_per_dim)

    common_plot_args = (xlabel="x1", ylabel="x2", aspect_ratio=:equal, colorbar=true, framestyle=:box)
    n_points_total = length(Y_history)
    obs_scatter_args = (marker_z=1:n_points_total, markersize=4, markerstrokewidth=0.5, markerstrokecolor=:white, colorbar_title="Iteration", label="", legend=false)
    best_scatter_args = (label="Best Found", color=:lime, markersize=8, marker=:star5, markerstrokecolor=:black)

    p_mean = contourf(x1_range, x2_range, μ_grid; title="GP Predicted Mean", common_plot_args...)
    scatter!(p_mean, X_history[1, :], X_history[2, :]; obs_scatter_args...)
    if best_x !== nothing
        scatter!(p_mean, [best_x[1]], [best_x[2]]; best_scatter_args...)
    end
    push!(plot_list, p_mean)

    p_std = contourf(x1_range, x2_range, σ_grid; title="GP Predicted Std Dev", common_plot_args...)
    scatter!(p_std, X_history[1, :], X_history[2, :]; obs_scatter_args...)
    if best_x !== nothing
        scatter!(p_std, [best_x[1]], [best_x[2]]; best_scatter_args...)
    end
    push!(plot_list, p_std)

    if objective_func !== nothing
        try
            Z_true = [objective_func([x1, x2]) for x2 in x2_range, x1 in x1_range]
            p_true = contourf(x1_range, x2_range, Z_true; title="True Objective", common_plot_args...)
            scatter!(p_true, X_history[1, :], X_history[2, :]; obs_scatter_args...)
            if best_x !== nothing
                scatter!(p_true, [best_x[1]], [best_x[2]]; best_scatter_args...)
            end
            push!(plot_list, p_true)
        catch e
            println("Warn: Could not plot 2D objective_func: $e")
        end
    end

    return plot_list
end

function _plot_2d_rf_mean(X_history, Y_history, final_model, bounds, objective_func, contour_points_per_dim)
    plot_list = []
    x1_range = range(bounds[1, 1], stop=bounds[1, 2], length=contour_points_per_dim)
    x2_range = range(bounds[2, 1], stop=bounds[2, 2], length=contour_points_per_dim)
    X_plot_mat = generate_grid(bounds, contour_points_per_dim)

    μ_plot, _ = RandomForestModel.predict_rf_mean_variance(final_model, X_plot_mat)
    μ_grid = reshape(μ_plot, contour_points_per_dim, contour_points_per_dim)

    common_plot_args = (xlabel="x1", ylabel="x2", aspect_ratio=:equal, colorbar=true, framestyle=:box)
    n_points_total = length(Y_history)
    obs_scatter_args = (marker_z=1:n_points_total, markersize=4, markerstrokewidth=0.5, markerstrokecolor=:white, colorbar_title="Iteration", label="", legend=false)

    p_mean = contourf(x1_range, x2_range, μ_grid; title="RF Predicted Mean", common_plot_args...)
    scatter!(p_mean, X_history[1, :], X_history[2, :]; obs_scatter_args...)
    push!(plot_list, p_mean)

    if objective_func !== nothing
        try
            Z_true = [objective_func([x1, x2]) for x2 in x2_range, x1 in x1_range]
            p_true = contourf(x1_range, x2_range, Z_true; title="True Objective", common_plot_args...)
            scatter!(p_true, X_history[1, :], X_history[2, :]; obs_scatter_args...)
            push!(plot_list, p_true)
        catch e
            println("Warn: Could not plot 2D objective_func: $e")
        end
    end

    return plot_list
end

function _plot_acquisition(final_gp, bounds, X_history, Y_history, acquisition_func, maximize, acq_func_params, contour_points_per_dim)
    dims = size(bounds, 1)
    if isempty(Y_history)
        println("Warn: Cannot plot acquisition function, history is empty.")
        return nothing
    end
    y_best_final = get_incumbent(Y_history, maximize)

    if dims == 1
        p = plot(title="Acquisition Function ($acquisition_func, Final Iter)", xlabel="x", ylabel="Acquisition Value", framestyle=:box)
        x_range = range(bounds[1, 1], stop=bounds[1, 2], length=200)
        x_plot_mat = reshape(collect(x_range), 1, length(x_range))
        if final_gp !== nothing && isa(final_gp, GPE)
            acq_values = calculate_acquisition(final_gp, x_plot_mat, acquisition_func, maximize, y_best_final; ξ=acq_func_params.ξ, κ=acq_func_params.κ)
            plot!(p, x_range, acq_values, label=string(acquisition_func), color=:purple, lw=2)
            vline!(p, [X_history[1, end]], label="Last Sampled", color=:orange, linestyle=:dash)
            return p
        else
            println("Warn: Cannot plot 1D acquisition, invalid GP model.")
            return nothing
        end
    elseif dims == 2
        x1_range = range(bounds[1, 1], stop=bounds[1, 2], length=contour_points_per_dim)
        x2_range = range(bounds[2, 1], stop=bounds[2, 2], length=contour_points_per_dim)
        X_plot_mat = generate_grid(bounds, contour_points_per_dim)
        if final_gp !== nothing && isa(final_gp, GPE)
            acq_values = calculate_acquisition(final_gp, X_plot_mat, acquisition_func, maximize, y_best_final; ξ=acq_func_params.ξ, κ=acq_func_params.κ)
            acq_grid = reshape(acq_values, contour_points_per_dim, contour_points_per_dim)
            common_plot_args = (xlabel="x1", ylabel="x2", aspect_ratio=:equal, colorbar=true, framestyle=:box)
            n_points_total = length(Y_history) # Use length
            obs_scatter_args = (marker_z=1:n_points_total, markersize=4, markerstrokewidth=0.5, markerstrokecolor=:white, colorbar_title="Iteration", label="", legend=false)

            p = contourf(x1_range, x2_range, acq_grid; title="Acquisition Function ($acquisition_func, Final Iter)", common_plot_args...)
            scatter!(p, X_history[1, :], X_history[2, :]; obs_scatter_args...)
            scatter!(p, [X_history[1, end]], [X_history[2, end]], label="Last Sampled", color=:orange, markersize=8, marker=:diamond, markerstrokecolor=:black)
            return p
        else
            println("Warn: Cannot plot 2D acquisition, invalid GP model.")
            return nothing
        end
    else
        println("Info: Acquisition function plot only supported for 1D/2D.")
        return nothing
    end
end

function _plot_convergence(X_history, Y_history, maximize, n_initial_points)
    dims = size(X_history, 1)
    n_points_total = length(Y_history)
    if n_points_total == 0
        return nothing
    end
    best_y_so_far = Vector{Float64}(undef, n_points_total)
    for i in 1:n_points_total
        best_y_so_far[i] = get_incumbent(Y_history[1:i], maximize)
    end
    p_conv = plot(1:n_points_total, best_y_so_far, xlabel="Number of Evaluations", ylabel="Best Objective Value Found", title="Convergence Plot (Dim = $dims)", label="Best Value", lw=2, legend=:topright, framestyle=:box)
    scatter!(p_conv, 1:n_points_total, Y_history, label="Observed Values", alpha=0.5, markersize=3)
    if n_initial_points !== nothing && n_initial_points > 0 && n_initial_points < n_points_total
        vline!(p_conv, [n_initial_points + 0.5], label="End of Initial Design", linestyle=:dash, color=:gray)
    end
    return p_conv
end

function _plot_observations_high_d(X_history, Y_history, n_initial_points)
    dims = size(X_history, 1)
    n_points_total = length(Y_history)
    if n_points_total == 0
        return nothing
    end
    p_obs = plot(1:n_points_total, Y_history, seriestype=:scatter, xlabel="Number of Evaluations", ylabel="Observed Objective Value", title="Observations Plot (Dim = $dims)", label="Observed Values", legend=:best, alpha=0.6, framestyle=:box)
    if n_initial_points !== nothing && n_initial_points > 0 && n_initial_points < n_points_total
        vline!(p_obs, [n_initial_points + 0.5], label="End of Initial Design", linestyle=:dash, color=:gray)
    end
    return p_obs
end

function _plot_parity(Y_actual::AbstractVector, Y_pred::AbstractVector, model_name::String, data_label::String, title_suffix::String)
    if isempty(Y_actual) || isempty(Y_pred) || length(Y_actual) != length(Y_pred)
        println("Warning: Cannot create parity plot for empty or mismatched data ($data_label).")
        return nothing
    end
    valid_idx = findall(isfinite.(Y_actual) .& isfinite.(Y_pred))
    if isempty(valid_idx)
        println("Warning: No valid finite data points for parity plot ($data_label).")
        return nothing
    end
    Y_actual_filt = Y_actual[valid_idx]
    Y_pred_filt = Y_pred[valid_idx]

    residuals = Y_actual_filt .- Y_pred_filt
    ss_res = sum(residuals .^ 2)
    mean_actual = mean(Y_actual_filt)
    ss_tot = sum((Y_actual_filt .- mean_actual) .^ 2)
    r_squared = ss_tot < 1e-12 ? (ss_res < 1e-12 ? 1.0 : 0.0) : (1.0 - (ss_res / ss_tot))
    rmse = sqrt(mean(residuals .^ 2))

    plot_title = "$model_name Parity Plot ($data_label)" * (isempty(title_suffix) ? "" : " - $title_suffix")
    plot_title *= @sprintf("\nR² = %.3f, RMSE = %.3e", r_squared, rmse)

    p = scatter(Y_actual_filt, Y_pred_filt,
        xlabel="Actual $data_label Y",
        ylabel="$model_name Predicted $data_label Y",
        title=plot_title,
        label="$data_label Set Predictions",
        markersize=4, markerstrokewidth=0, alpha=0.7,
        legend=:bottomright,
        aspect_ratio=:equal,
        framestyle=:box)

    min_val = min(minimum(Y_actual_filt), minimum(Y_pred_filt))
    max_val = max(maximum(Y_actual_filt), maximum(Y_pred_filt))
    range_padding = (max_val - min_val) * 0.05
    if max_val ≈ min_val
        range_padding = max(abs(min_val * 0.1), 0.1)
    end
    diag_range = (min_val - range_padding, max_val + range_padding)
    plot!(p, [diag_range[1], diag_range[2]], [diag_range[1], diag_range[2]],
        label="y = x (Parity Line)",
        color=:black, linestyle=:dash, lw=2)

    println("$model_name Parity Plot ($data_label) generated. R² = $(round(r_squared, digits=3)), RMSE = $(round(rmse, digits=3))")
    return p
end

function _plot_test_rmse_history(test_rmse_history, n_initial_points)
    if test_rmse_history === nothing || isempty(test_rmse_history)
        println("Info: No test RMSE history found. Cannot plot test performance.")
        return nothing
    end
    valid_indices = findall(!isnan, test_rmse_history)
    if isempty(valid_indices)
        println("Warn: Test RMSE history contains only NaNs. Cannot plot.")
        return nothing
    end
    valid_rmse = test_rmse_history[valid_indices]
    if n_initial_points === nothing
        n_initial_points = 0
        println("Warning: Missing n_initial_points for RMSE plot. Assuming 0.")
    end
    total_training_points_valid = (n_initial_points) .+ valid_indices

    println("Generating AL Test Set Performance Plot...")
    p = plot(total_training_points_valid, valid_rmse, # Plot only valid points
        xlabel="Number of Training Points",
        ylabel="Test Set RMSE",
        title="Active Learning: Test Set RMSE vs Training Set Size",
        label="Test RMSE", lw=2, legend=:topright, framestyle=:box)
    if n_initial_points > 0
        vline!(p, [n_initial_points + 0.5], label="Start of AL", linestyle=:dash, color=:gray)
    end
    return p
end

"""
    plot_results(results; ...)

Generates a combined plot visualizing results from Bayesian Optimization or Active Learning.

Automatically detects dimensionality, model type (GP or RF), and available data
(e.g., test sets) in the `results` object (NamedTuple or Dict).

Args:
    results: A NamedTuple or Dict containing the output from `bayes_optimize`,
             `active_learn`, `active_learn_rf`, or `active_learn_residual_rf_gp`. Must contain at least
             `:X_history` and `:Y_history`.

Keyword Args:
    objective_func: The true objective function (optional, for plotting ground truth).
    plot_model_fit::Bool (true): Plot the surrogate model's predictions vs. observations (1D/2D).
    plot_acquisition::Bool (false): Plot the acquisition function (BO, GP, 1D/2D only).
    plot_convergence_or_high_d::Bool (true): Plot convergence (BO >2D) or observations (>2D AL).
    plot_parity_train::Bool (true): Plot model parity on training data.
    plot_parity_test::Bool (true): Plot model parity on test data (if available).
    plot_test_performance::Bool (true): Plot test RMSE history (AL, if available).
    contour_points_per_dim::Int (50): Resolution for 2D contour plots.
    acq_func_params: Parameters (ξ, κ) for acquisition function plots (defaults to (ξ=0.01, κ=2.0)).
    title_suffix::String (""): Suffix to add to parity plot titles.

Returns:
    A Plots.Plot object containing the combined visualization, or `nothing` if no plots
    could be generated.
"""
function plot_results(results;
    objective_func=nothing,
    plot_model_fit::Bool=true,
    plot_acquisition::Bool=false,
    plot_convergence_or_high_d::Bool=true,
    plot_parity_train::Bool=true,
    plot_parity_test::Bool=true,
    plot_test_performance::Bool=true,
    contour_points_per_dim::Int=50,
    acq_func_params=(ξ=0.01, κ=2.0),
    title_suffix::String="",
    plot_size = "auto"
)

    plot_list = []

    X_history = _safeget(results, :X_history)
    Y_history = _safeget(results, :Y_history)
    bounds = _safeget(results, :bounds)
    model_type_flag = _safeget(results, :model_type)

    model_name = "Unknown"
    model_type = :Unknown

    if model_type_flag == :GP
        final_gp = _safeget(results, :final_gp)
        if final_gp !== nothing && isa(final_gp, GPE)
            model_type = :GP
            model_name = "GP"
            println("Detected Model Type: GP")
        else
            println("Warning: model_type is :GP but :final_gp is invalid.")
        end
    elseif model_type_flag == :RandomForest
        final_model_rf = _safeget(results, :final_model)
        if final_model_rf !== nothing && isa(final_model_rf, RandomForestModel.RandomForestRegressor)
            model_type = :RandomForest
            model_name = "RF"
            println("Detected Model Type: RandomForest")
        else
            println("Warning: model_type is :RandomForest but :final_model is invalid or wrong type. Got: ", typeof(final_model_rf))
        end
    elseif model_type_flag == :ResidualRFGP
        final_rf = _safeget(results, :final_rf_model)
        final_gp_res = _safeget(results, :final_residual_gp)
        if final_rf !== nothing && final_gp_res !== nothing && isa(final_gp_res, GPE) && isa(final_rf, RandomForestModel.RandomForestRegressor)
            model_type = :ResidualRFGP
            model_name = "RF+GP"
            println("Detected Model Type: ResidualRFGP")
        else
            println("Warning: model_type is :ResidualRFGP but required models (:final_rf_model, :final_residual_gp) are missing or invalid.")
        end
    else
        println("Warning: Could not determine valid model type from results (:model_type flag was '", model_type_flag, "').")
    end

    best_x = _safeget(results, :best_x)
    best_y = _safeget(results, :best_y)
    maximize = _safeget(results, :maximize, false)
    acquisition_func = _safeget(results, :acquisition_func)
    n_iterations = _safeget(results, :n_iterations)
    n_initial_points = _safeget(results, :n_initial_points)
    X_test = _safeget(results, :X_test)
    Y_test = _safeget(results, :Y_test)
    test_rmse_history = _safeget(results, :test_rmse_history)

    if n_initial_points === nothing && n_iterations !== nothing && X_history !== nothing
        n_initial_points = max(0, size(X_history, 2) - n_iterations)
    end

    if X_history === nothing || Y_history === nothing
        println("Error: History data (X_history, Y_history) missing. Cannot generate plots.")
        return nothing
    end
    input_dim = size(X_history, 1)
    if input_dim == 0
        println("Error: Input dimension is 0.")
        return nothing
    end
    if bounds === nothing && input_dim <= 2 && (plot_model_fit || plot_acquisition)
        println("Warning: Bounds data missing, cannot generate 1D/2D model or acquisition plots.")
    end

    println("Generating visualization plots (Input Dim: $input_dim, Model: $model_name)...")
    if plot_model_fit && model_type != :Unknown && input_dim <= 2 && bounds !== nothing
        println(" -> Plotting model fit...")
        local p_fit = nothing
        if input_dim == 1
            if model_type == :GP
                gp_model = _safeget(results, :final_gp)
                if gp_model !== nothing && isa(gp_model, GPE)
                    p_fit = _plot_1d_gp(X_history, Y_history, gp_model, bounds, objective_func, best_x, best_y)
                else
                    println("Warning: Cannot plot 1D GP fit, :final_gp missing or invalid.")
                end
            elseif model_type == :RandomForest
                rf_model = _safeget(results, :final_model)
                if rf_model !== nothing && isa(rf_model, RandomForestModel.RandomForestRegressor)
                    p_fit = _plot_1d_rf(X_history, Y_history, rf_model, bounds, objective_func)
                else
                    println("Warning: Cannot plot 1D RF fit, :final_model missing or invalid.")
                end
            elseif model_type == :ResidualRFGP
                println(" -> Plotting detailed 1D model fit for ResidualRFGP model type...") # Changed message
                final_rf = _safeget(results, :final_rf_model)
                final_gp_res = _safeget(results, :final_residual_gp)
                p_fit = _plot_1d_residual_rf_gp(X_history, Y_history, final_rf, final_gp_res, bounds, objective_func)
            end
            if p_fit !== nothing
                push!(plot_list, p_fit)
            end

        elseif input_dim == 2
            local p_list_2d = []
            if model_type == :GP
                gp_model = _safeget(results, :final_gp)
                if gp_model !== nothing && isa(gp_model, GPE)
                    p_list_2d = _plot_2d_gp(X_history, Y_history, gp_model, bounds, objective_func, best_x, contour_points_per_dim)
                else
                    println("Warning: Cannot plot 2D GP contours, :final_gp missing or invalid.")
                end
            elseif model_type == :RandomForest
                rf_model = _safeget(results, :final_model)
                if rf_model !== nothing && isa(rf_model, RandomForestModel.RandomForestRegressor)
                    p_list_2d = _plot_2d_rf_mean(X_history, Y_history, rf_model, bounds, objective_func, contour_points_per_dim)
                else
                    println("Warning: Cannot plot 2D RF contours, :final_model missing or invalid.")
                end
            elseif model_type == :ResidualRFGP
                println(" -> Skipping detailed 2D model fit plot for ResidualRFGP model type (parity plots are available).")
            end
            if !isempty(p_list_2d)
                append!(plot_list, p_list_2d)
            end
        end
    end
    if plot_acquisition && model_type == :GP && acquisition_func !== nothing && input_dim <= 2 && bounds !== nothing
        println(" -> Plotting acquisition function...")
        gp_model_for_acq = _safeget(results, :final_gp)
        if gp_model_for_acq !== nothing && isa(gp_model_for_acq, GPE)
            p = _plot_acquisition(gp_model_for_acq, bounds, X_history, Y_history, acquisition_func, maximize, acq_func_params, contour_points_per_dim)
            if p !== nothing
                push!(plot_list, p)
            end
        else
            println("Warning: Cannot plot acquisition function, :final_gp missing or invalid.")
        end
    end

    if plot_convergence_or_high_d && input_dim > 2
        is_bo_result = best_x !== nothing
        if is_bo_result
            println(" -> Plotting convergence...")
            p = _plot_convergence(X_history, Y_history, maximize, n_initial_points)
            if p !== nothing
                push!(plot_list, p)
            end
        else
            println(" -> Plotting high-dimensional observations...")
            p = _plot_observations_high_d(X_history, Y_history, n_initial_points)
            if p !== nothing
                push!(plot_list, p)
            end
        end
    end

    if plot_parity_train && model_type != :Unknown
        println(" -> Plotting training parity...")
        local Y_pred_train = []
        plot_model_name = model_name

        try
            if model_type == :GP
                gp_model = _safeget(results, :final_gp)
                if gp_model !== nothing && isa(gp_model, GPE)
                    Y_pred_train, _ = predict_with_gp(gp_model, X_history)
                end
            elseif model_type == :RandomForest
                rf_model = _safeget(results, :final_model)
                if rf_model !== nothing && isa(rf_model, RandomForestModel.RandomForestRegressor)
                    Y_pred_train, _ = RandomForestModel.predict_rf_mean_variance(rf_model, X_history)
                end
            elseif model_type == :ResidualRFGP
                final_rf = _safeget(results, :final_rf_model)
                final_gp = _safeget(results, :final_residual_gp)
                plot_model_name = "RF+GP"
                if final_rf !== nothing && final_gp !== nothing
                    rf_preds, _ = RandomForestModel.predict_rf_mean_variance(final_rf, X_history)
                    gp_preds, _ = predict_with_gp(final_gp, X_history)
                    Y_pred_train = rf_preds .+ gp_preds
                end
            end
        catch e
            println("Warning: Error during prediction for training parity plot ($model_type): $e")
            Y_pred_train = []
        end

        if !isempty(Y_pred_train) && !isempty(Y_history)
            p = _plot_parity(Y_history, Y_pred_train, plot_model_name, "Training", title_suffix)
            if p !== nothing
                push!(plot_list, p)
            end
        elseif model_type != :Unknown
            println("Warning: Could not generate training parity plot for $model_name (predictions failed or model invalid).")
        end
    end
    if plot_parity_test && model_type != :Unknown && X_test !== nothing && Y_test !== nothing
        println(" -> Plotting test parity...")
        local Y_pred_test = []
        plot_model_name = model_name

        try
            if model_type == :GP
                gp_model = _safeget(results, :final_gp)
                if gp_model !== nothing && isa(gp_model, GPE)
                    Y_pred_test, _ = predict_with_gp(gp_model, X_test)
                end
            elseif model_type == :RandomForest
                rf_model = _safeget(results, :final_model)
                if rf_model !== nothing && isa(rf_model, RandomForestModel.RandomForestRegressor)
                    Y_pred_test, _ = RandomForestModel.predict_rf_mean_variance(rf_model, X_test)
                end
            elseif model_type == :ResidualRFGP
                final_rf = _safeget(results, :final_rf_model)
                final_gp = _safeget(results, :final_residual_gp)
                plot_model_name = "RF+GP"
                if final_rf !== nothing && final_gp !== nothing
                    rf_preds, _ = RandomForestModel.predict_rf_mean_variance(final_rf, X_test)
                    gp_preds, _ = predict_with_gp(final_gp, X_test)
                    Y_pred_test = rf_preds .+ gp_preds
                end
            end
        catch e
            println("Warning: Error during prediction for test parity plot ($model_type): $e")
            Y_pred_test = []
        end

        if !isempty(Y_pred_test) && !isempty(Y_test)
            p = _plot_parity(Y_test, Y_pred_test, plot_model_name, "Test", title_suffix)
            if p !== nothing
                push!(plot_list, p)
            end
        elseif model_type != :Unknown
            println("Warning: Could not generate test parity plot for $model_name (predictions failed or model invalid).")
        end
    end
    if plot_test_performance && test_rmse_history !== nothing
        println(" -> Plotting test performance history...")
        p = _plot_test_rmse_history(test_rmse_history, n_initial_points)
        if p !== nothing
            push!(plot_list, p)
        end
    end
    if isempty(plot_list)
        println("No plots generated based on available data and selected options.")
        return nothing
    end

    n_plots = length(plot_list)
    layout_cols = ceil(Int, sqrt(n_plots))
    layout_rows = ceil(Int, n_plots / layout_cols)
    plot_width = max(600, 450 * layout_cols)
    plot_height = max(400, 350 * layout_rows)
    if plot_size == "auto"
    plot_size = (plot_width, plot_height)
    end
    println("Combining $n_plots generated plot(s) into a layout of ($layout_rows, $layout_cols)...")
    final_plot = plot(plot_list..., layout=(layout_rows, layout_cols), size=plot_size)

    return final_plot

end


end