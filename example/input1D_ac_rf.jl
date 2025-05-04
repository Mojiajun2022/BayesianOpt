using BayesianOpt 
using Plots
using Random
using Printf 

function complex_1d_func(x_vec::AbstractVector)
    x = x_vec[1]
    term1 = sin(x * 10.0)
    term2 = exp(-0.5 * ((x - 0.6)/0.1)^2) 
    term3 = exp(-0.5 * ((x - 0.2)/0.05)^2) 
    return term1 + 0.5*term2 + 0.3*term3 + randn()*0.05
end

n_al_iterations = 100
n_initial_al = 200   
seed_al = 5678

bounds_1d = reshape([0.0, 2.0], 1, 2) 

rf_params = Dict(
    :max_depth => -1,           
    :min_samples_leaf => 3,    
    :min_samples_split => 2,   
    :partial_sampling => 0.7   
    # :n_subfeatures => -1     
)
println("Random Forest Hyperparameters (rf_kwargs): ", rf_params)
X_test_mat = BayesianOpt.Utils.sample_randomly(bounds_1d, n_test_points)
Y_test_vec = [complex_1d_func_noiseless(X_test_mat[:, i]) for i in 1:n_test_points]
println("\nStart running active learning (random forest)...")
Random.seed!(seed_al)
al_rf_results = active_learn_rf(
    complex_1d_func,         
    bounds_1d,               
    n_al_iterations,         
    n_initial_points=n_initial_al,
    variance_maximizer=LBFGS, 
    X_test=X_test_mat,       
    Y_test=Y_test_vec,       
    rf_n_trees=1000,          
    rf_kwargs=rf_params,     
    verbose=true,
)

println("\n--- Active Learning Results (Random Forest) ---")
if @isdefined(al_rf_results) && al_rf_results !== nothing
    println("Final sample count: ", size(al_rf_results.X_history, 2))
    if hasproperty(al_rf_results, :final_model)
        println("Final model type: ", typeof(al_rf_results.final_model))
    else
         println("The `final_model` field was not found in the results.")
    end
else
    println("Error: Active learning failed to return results successfully.")
end


println("\nGenerating Active Learning Result Visualization (Random Forest)...")

plot_results(al_rf_results)