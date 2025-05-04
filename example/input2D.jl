using BayesianOpt
using Plots       
using Random      
gr(size=(1000, 800))

function branin_function(x::AbstractVector{<:Real})
    if length(x) != 2
        error("Branin function requires a 2-dimensional input vector.")
    end
    x1_mapped = x[1] * 15.0 - 5.0  
    x2_mapped = x[2] * 15.0        

    a = 1.0
    b = 5.1 / (4.0 * π^2)
    c = 5.0 / π
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * π)

    term1 = a * (x2_mapped - b * x1_mapped^2 + c * x1_mapped - r)^2
    term2 = s * (1 - t) * cos(x1_mapped)
    result = term1 + term2 + s

    return result
end

n_test_points = 200
bounds = [0.0 1.0;  
          0.0 1.0]  
X_test = BayesianOpt.Utils.sample_randomly(bounds, n_test_points)
Y_test = [branin_function(X_test[:, i]) for i in 1:n_test_points]
total_iterations_bo = 40  
total_iterations_al = 40  
initial_points = 8      
common_seed = 1234      

println("--- Bayesian Optimization Example (2D Branin Function) ---")

Random.seed!(common_seed)
bo_results = bayes_optimize(
    branin_function,      
    bounds,               
    total_iterations_bo,  
    n_initial_points=initial_points,
    initial_design=:random,
    acquisition_func=EI,   
    acq_maximizer=LBFGS,   
    maximize=false,        
    verbose=true
    # random_seed=common_seed # Seed is set externally or can be passed here
)

println("\n--- Bayesian Optimization Final Results ---")
println("Best point found (in [0,1]^2 space): x_best = ", round.(bo_results.best_x, digits=4))
# Map best_x back to Branin domain for context
x1_best_mapped = bo_results.best_x[1] * 15.0 - 5.0
x2_best_mapped = bo_results.best_x[2] * 15.0
println("Best point found (in Branin domain):    x_map = [", round(x1_best_mapped, digits=4), ", ", round(x2_best_mapped, digits=4), "]")
println("Best objective value found:           y_best = ", round(bo_results.best_y, digits=6))
println("Known minimum is approx 0.397887")

# --- 7. Visualize Bayesian Optimization Results ---
println("\nGenerating Bayesian Optimization visualization...")
plot_results(bo_results, )