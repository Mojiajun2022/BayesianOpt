using BayesianOpt
using Plots
using Random
using LinearAlgebra
using Statistics
using Printf
using GaussianProcesses
using Distributions

gr()

function complex_1d_func_noisy(x_vec::AbstractVector)
    x = x_vec[1]
    term1 = sin(x * 10.0)
    term2 = exp(-0.5 * ((x - 0.6)/0.1)^2) 
    term3 = exp(-0.5 * ((x - 0.2)/0.05)^2)
    noise = randn() * 0.03 
    return term1 + 0.5*term2 + 0.3*term3 + noise
end

function complex_1d_func_noiseless(x_vec::AbstractVector)
    x = x_vec[1]
    term1 = sin(x * 10.0)
    term2 = exp(-0.5 * ((x - 0.6)/0.1)^2)
    term3 = exp(-0.5 * ((x - 0.2)/0.05)^2)
    return term1 + 0.5*term2 + 0.3*term3
end


bounds_1d = [0.0 1.0] 
println("Search boundary: ", bounds_1d)

println("Creating a fixed test set for evaluation...")
n_test_points = 200      
Random.seed!(123)       
X_test_mat = BayesianOpt.Utils.sample_randomly(bounds_1d, n_test_points)
Y_test_vec = [complex_1d_func_noiseless(X_test_mat[:, i]) for i in 1:n_test_points]
println("Generated a test set with $n_test_points points using the noiseless function.")

n_al_iterations = 100
n_initial_al = 10    
seed_al = 5678

rf_n_trees_hybrid = 2000
rf_params_hybrid = Dict(
    :min_samples_leaf => 3,
    :max_depth => -1
)
println("RF Parameters (rf_kwargs): ", rf_params_hybrid)

kernel_gp_res = SEIso(0.0, 0.0)
noise_prior = Normal(-6.0, 1.5) 
println("GP logNoise priori: ", noise_prior)

maximizer_choice = LBFGS
maximizer_config = Dict(:lbfgs_starts => 15)
println("\nStart running hybrid RF+GP active learning (including test set)...")
Random.seed!(seed_al)
al_results_hybrid = active_learn_residual_rf_gp(
    complex_1d_func_noisy, 
    bounds_1d,
    n_al_iterations,
    n_initial_points=n_initial_al,
    X_test=X_test_mat,       
    Y_test=Y_test_vec,       
    rf_n_trees=rf_n_trees_hybrid,
    rf_kwargs=rf_params_hybrid,
    kernel=kernel_gp_res,
    logNoise_prior=noise_prior,
    optimize_gp_hypers=true,
    variance_maximizer=maximizer_choice,
    maximizer_kwargs=maximizer_config,
    verbose=true,
    random_seed=seed_al
)

println("\n--- Mix RF+GP Active Learning Results ---")
println("Final sample count: ", size(al_results_hybrid.X_history, 2))
if haskey(al_results_hybrid, :test_rmse_history) && !isempty(al_results_hybrid.test_rmse_history)
    valid_rmse = filter(!isnan, al_results_hybrid.test_rmse_history)
    if !isempty(valid_rmse)
        println("Final test set RMSE (compared to noise-free ground truth): ", round(valid_rmse[end], digits=4))
    else
        println("Test set RMSE history invalid.")
    end
end

println("\nGenerate visualizations of hybrid RF+GP active learning results....")
results_plot = plot_results(al_results_hybrid,
                            objective_func = complex_1d_func_noiseless, 
                            plot_model_fit=true, 
                            plot_parity_train=true,
                            plot_parity_test=true,
                            plot_test_performance=true 
                           )

if results_plot !== nothing
    display(results_plot)
else
    println("plot_results did not return a plot object.")
end
