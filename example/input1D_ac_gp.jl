using BayesianOpt

function complex_1d_func(x_vec::AbstractVector)
    x = x_vec[1]
    term1 = sin(x * 10.0)
    term2 = exp(-0.5 * ((x - 0.6)/0.1)^2) 
    term3 = exp(-0.5 * ((x - 0.2)/0.05)^2) 
    return term1 + 0.5*term2 + 0.3*term3 + randn()*0.05 
end

bounds_1d = [0.0 1.0] 

n_al_iterations = 50  
n_initial_al = 3     
seed_al = 5678
n_test_points = 200      
Random.seed!(123)       
X_test_mat = BayesianOpt.Utils.sample_randomly(bounds_1d, n_test_points)
Y_test_vec = [complex_1d_func_noiseless(X_test_mat[:, i]) for i in 1:n_test_points]

println("Start running active learning....")
al_results = active_learn(
    complex_1d_func,      
    bounds_1d,
    X_test=X_test_mat,       
    Y_test=Y_test_vec,                   
    n_al_iterations,      
    n_initial_points=n_initial_al,
    variance_maximizer=LBFGS,
    verbose=true,
    random_seed=seed_al,
    optimize_gp_hypers=true 
)

println("\n--- Active learning results ---")
println("Final sample count: ", size(al_results.X_history, 2))

println("\nGenerate active learning result visualization...")
using Plots
gr()


plot_results(al_results)