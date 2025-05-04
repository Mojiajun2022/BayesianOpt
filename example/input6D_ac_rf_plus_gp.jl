using BayesianOpt
using Plots
using Random
using LinearAlgebra
using Statistics
using Printf
using GaussianProcesses
using Distributions
gr()

# Defined on [0, 1]^6. It has a known global minimum.
# Source for constants: https://www.sfu.ca/~ssurjano/hart6.html

function hartmann6_function(x::AbstractVector{<:Real})
    if length(x) != 6
        error("Hartmann 6 function requires a 6-dimensional input vector.")
    end
    if any(xi < 0.0 || xi > 1.0 for xi in x)
         error("Input x = $x is outside the [0, 1]^6 domain for Hartmann 6.")
    end

    alpha = [1.0, 1.2, 3.0, 3.2]

    A = [10.0  3.0   17.0  3.5   1.7   8.0;
          0.05 10.0   17.0  0.1   8.0   14.0;
          3.0   3.5   1.7  10.0  17.0   8.0;
         17.0  8.0    0.05 10.0   0.1  14.0] # 4x6 matrix

    P = 1e-4 * [1312.0  1696.0  5569.0  124.0   8283.0  5886.0;
                2329.0  4135.0  8307.0  3736.0  1004.0  9991.0;
                2348.0  1451.0  3522.0  2883.0  3047.0  6650.0;
                4047.0  8828.0  8732.0  5743.0  1091.0   381.0] # 4x6 matrix

    outer_sum = 0.0
    for i in 1:4
        inner_sum = 0.0
        for j in 1:6
            inner_sum += A[i, j] * (x[j] - P[i, j])^2
        end
        outer_sum += alpha[i] * exp(-inner_sum)
    end

    # Known global minimum of the negated function is ≈ -3.32237
    return -outer_sum
end

bounds = zeros(6, 2)
bounds[:, 1] .= 0.0 
bounds[:, 2] .= 1.0 

println("Bounds:\n", bounds)

noise_prior = Normal(-6.0, 1.5)
println("Defined logNoise prior: ", noise_prior)
n_test_points = 100
X_test_mat = BayesianOpt.Utils.sample_randomly(bounds, n_test_points)
Y_test_vec = [hartmann6_function(X_test_mat[:, i]) for i in 1:n_test_points]
total_iterations_al = 200
initial_points = 1000
common_seed = 2025
rf_n_trees_hybrid = 2000 
rf_params_hybrid = Dict(    :max_depth => 15,         
:min_samples_leaf => 5,    
:partial_sampling => 0.7   
) 
kernel_seard_res = SEArd(fill(-1.0, 6), 0.0)
maximizer_choice = LBFGS 
maximizer_config = Dict(:lbfgs_starts => 50) 
Random.seed!(common_seed)
al_results_hybrid = active_learn_residual_rf_gp(
    hartmann6_function,
    bounds,
    total_iterations_al,
    n_initial_points=initial_points,
    X_test=X_test_mat,
    Y_test=Y_test_vec,
    initial_design=:random,
    rf_n_trees=rf_n_trees_hybrid,
    rf_kwargs=rf_params_hybrid,  
    kernel=kernel_seard_res,     
    logNoise_prior=noise_prior,  
    variance_maximizer=maximizer_choice, 
    maximizer_kwargs=maximizer_config,   
    verbose=false
)


println("\n--- RF Active Learning Final Results ---")
println("Total points sampled (training): ", size(al_results_hybrid.X_history, 2))
if haskey(al_results_hybrid, :test_rmse_history) && !isempty(al_results_hybrid.test_rmse_history)
    println("Final Test RMSE: ", round(al_results_hybrid.test_rmse_history[end], digits=4))
end


println("\nGenerating RF AL observations plot...")
al_rf_obs_plot = plot_results(al_results_hybrid) 
