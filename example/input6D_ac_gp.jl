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

# Example: Normal distribution centered around log(0.05^2) ≈ -6.0, std dev 1.5
noise_prior = Normal(-6.0, 1.5)
# println("Defined logNoise prior: ", noise_prior)

total_iterations_al = 500
initial_points = 100    
common_seed = 2025
maximizer_choice = LBFGS 
lbfgs_starts_count = 50  
kernel_seard = SEArd(fill(-1.0, 6), 0.0) 


println("\n\n--- Active Learning Example (LBFGS Maximizer with Starts & Prior) ---")
println("Iterations: $total_iterations_al, Initial Points: $initial_points")
println("Variance Maximizer: $maximizer_choice with $lbfgs_starts_count starts") # Updated print statement
println("Using logNoise Prior: $noise_prior")

Random.seed!(common_seed)

al_results_6d_test = active_learn(
    hartmann6_function,
    bounds,            
    total_iterations_al,
    n_initial_points=initial_points,
    X_test=X_test_mat,
    Y_test=Y_test_vec,
    initial_design=:random,
    variance_maximizer=maximizer_choice,
    maximizer_lbfgs_starts=lbfgs_starts_count, 
    # maximizer_kwargs=...,
    verbose=true,
    kernel=kernel_seard,
    logNoise_prior=noise_prior,
    # optimize_gp_hypers=true, # Default is true
    # logNoise=-2.0, # Initial logNoise value is still used by update_gp! if GP is created initially
)
plot_results(al_results_6d_test)