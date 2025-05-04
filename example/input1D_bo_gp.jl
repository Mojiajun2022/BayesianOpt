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

println("Start running active learning....")
al_results = bayes_optimize(
    complex_1d_func,      
    bounds_1d,            
    n_al_iterations,      
    n_initial_points=n_initial_al,
    acq_maximizer=LBFGS,
    verbose=true,
    random_seed=seed_al,
    optimize_gp_hypers=true,
    maximize =false
)

println("\n--- Active learning results ---")
println("Final sample count: ", size(al_results.X_history, 2))

println("\nGenerate active learning result visualization...")
using Plots
gr()


plot_results(al_results,plot_size = (1200,400))