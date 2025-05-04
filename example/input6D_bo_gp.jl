using BayesianOpt
using Plots
using Random
using LinearAlgebra
using Statistics       
using Printf           
using GaussianProcesses
using Distributions    

gr()

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
         17.0  8.0    0.05 10.0   0.1  14.0]
    P = 1e-4 * [1312.0  1696.0  5569.0  124.0   8283.0  5886.0;
                2329.0  4135.0  8307.0  3736.0  1004.0  9991.0;
                2348.0  1451.0  3522.0  2883.0  3047.0  6650.0;
                4047.0  8828.0  8732.0  5743.0  1091.0   381.0]
    outer_sum = 0.0
    for i in 1:4
        inner_sum = 0.0
        for j in 1:6
            inner_sum += A[i, j] * (x[j] - P[i, j])^2
        end
        outer_sum += alpha[i] * exp(-inner_sum)
    end
    return -outer_sum 
end

bounds = zeros(6, 2)
bounds[:, 1] .= 0.0
bounds[:, 2] .= 1.0
println("Bounds:\n", bounds)

println("Creating a fixed test set for final evaluation...")
n_test_points = 500       
Random.seed!(999)         
X_test_mat = BayesianOpt.Utils.sample_randomly(bounds, n_test_points)
Y_test_vec = [hartmann6_function(X_test_mat[:, i]) for i in 1:n_test_points]
println("Generated a test set with $n_test_points points.")

n_iterations_bo = 200
n_initial_points_bo = 100
common_seed = 3000

println("\n\n--- Bayesian Optimization Example (6D Hartmann Function) ---")

Random.seed!(common_seed)
kernel_seard = SEArd(fill(0.0, 6), 0.0)

bo_gp_results = bayes_optimize(
    hartmann6_function,
    bounds,
    n_iterations_bo,
    n_initial_points=n_initial_points_bo,
    initial_design=:random,
    kernel=kernel_seard,       
    logNoise=2.0,              
    optimize_gp_hypers=true,   
    # logNoise_prior=nothing,   # Add prior here if desired, e.g., using Distributions.Normal(-9.0, 1.0)
    acquisition_func=EI,        
    acq_maximizer=LBFGS,        
    maximizer_lbfgs_starts=50,  
    maximize=false,             
    verbose=true,
    random_seed=common_seed     
)

println("\n--- Bayesian Optimization Final Results ---")
println("Total points sampled (training set size): ", size(bo_gp_results.X_history, 2))
println("Best minimum value found (y): ", bo_gp_results.best_y)
println("Location of minimum (x): ", round.(bo_gp_results.best_x, digits=4))
println("Known global minimum is ≈ -3.32237")

println("\n--- Evaluating Final GP Model on Test Set ---")
final_gp = bo_gp_results.final_gp

test_rmse = NaN
test_r_squared = NaN

if final_gp !== nothing && isa(final_gp, GPE)
    mu_test_pred, sigma_test_pred = BayesianOpt.GPModel.predict_with_gp(final_gp, X_test_mat)

    test_residuals = Y_test_vec .- mu_test_pred
    test_rmse = sqrt(mean(test_residuals .^ 2))
    test_ss_res = sum(test_residuals .^ 2)
    test_ss_tot = sum((Y_test_vec .- mean(Y_test_vec)).^ 2)
    test_r_squared = test_ss_tot < 1e-12 ? (test_ss_res < 1e-12 ? 1.0 : 0.0) : (1.0 - (test_ss_res / test_ss_tot))


    @printf("Test Set RMSE: %.4e\n", test_rmse)
    @printf("Test Set R²:   %.4f\n", test_r_squared)
else
    println("Could not evaluate on test set: Final GP model is missing or invalid in results.")
end

println("\nGenerating visualization plots...")

plot_data = merge(
    bo_gp_results, 
    (X_test = X_test_mat, Y_test = Y_test_vec) # Add test set explicitly
)

results_plot = plot_results(plot_data,
                            # You can explicitly turn plots on/off here if needed:
                            # plot_model_fit=false, # No 1D/2D model fit plot for 6D
                            # plot_acquisition=false, # No acquisition plot for 6D
                            plot_convergence_or_high_d=true, # Show convergence
                            plot_parity_train=true, # Show training parity
                            plot_parity_test=true, # Show test parity
                            # plot_test_performance=false # No test RMSE *history* from BO
                           )

if results_plot !== nothing
    display(results_plot)
else
    println("Plot generation failed or produced no plots.")
end

println("\n--- BO Example Script with Test Set Evaluation Finished ---")