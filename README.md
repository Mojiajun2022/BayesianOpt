# BayesianOpt (A simple program, for personal study only)
A Julia program for Bayesian optimization and active learning. The entire calculation is based on Gaussian Processes.jl and DecisionTree.jl.
The program can perform Gaussian process-based Bayesian optimization, as well as active learning algorithms based on Gaussian processes (GP) or random forests (RF). It is even possible to mix GP and RF to perform active learning for any parameter dimension. This calculation can also define the prior for GP noise.


# Bayesian Optimization Example

This section demonstrates a simple application of the `BayesianOpt.jl` package to optimize a 1D synthetic function.Objective Function

    using BayesianOpt
Definite a function

    function complex_1d_func(x_vec::AbstractVector)
      x = x_vec[1]
      term1 = sin(x * 10.0)
      term2 = exp(-0.5 * ((x - 0.6)/0.1)^2)
      term3 = exp(-0.5 * ((x - 0.2)/0.05)^2)
      
      return term1 + 0.5*term2 + 0.3*term3 + randn()*0.05
    end
    
This Julia function defines the target we want to optimize. It's a one-dimensional function (`x_vec[1]`) designed to have multiple local minima/maxima due to the sine term and two Gaussian peaks, plus a small amount of random noise (`randn()*0.05`) to simulate real-world uncertainty. The Bayesian Optimization algorithm will try to find the input value `x` that results in the minimum output value of this function within a specified range.Search Space Bounds
    
    bounds_1d = [0.0 1.0]
This line defines the boundaries of the search space for our 1D function. The optimizer will only consider `x` values between 0.0 and 1.0 (inclusive) when searching for the optimum.Optimization Parameters:
    
    n_al_iterations = 50
    n_initial_al = 3
    seed_al = 5678
These variables set key parameters for the optimization process:

`n_al_iterations`: The total number of iterations (function evaluations) the Bayesian Optimization algorithm will perform after the initial random sampling.

`n_initial_al`: The number of points that will be sampled randomly within the `bounds_1d` before the main Bayesian Optimization loop begins. These initial points help build the first Gaussian Process model.

`seed_al`: A seed for the random number generator. Using a fixed seed ensures that the sequence of random numbers (used for initial points and potentially noise) is the same every time the code runs, making the results reproducible.Running Bayesian Optimization
    
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
This is the main function call that executes the Bayesian Optimization.

`complex_1d_func`: The objective function to be optimized.

`bounds_1d`: The search space defined earlier.

`n_al_iterations`: The number of optimization steps.

`n_initial_points=n_initial_al`: Specifies the number of initial random points.

`acq_maximizer=LBFGS`: Sets the method used to find the maximum of the acquisition function (which determines the next point to sample). L-BFGS is a common gradient-based optimization algorithm.

`verbose=true`: Enables detailed output during the optimization process, showing the progress at each iteration.`random_seed=seed_al`: Sets the random seed for the optimization run itself, ensuring reproducibility.

`optimize_gp_hypers=true`: Instructs the algorithm to optimize the hyperparameters of the internal Gaussian Process model at each iteration, which can improve the model's accuracy.

`maximize=false`: Specifies that the goal is to find the minimum of the function. If set to `true`, it would search for the maximum.Viewing Results

    println("Final sample count: ", size(al_results.X_history, 2))
    using Plots
    gr()
    plot_results(al_results,plot_size = (1200,400))

`plot_results(al_results)`: Uses a function `plot_results(al_results)` to generate a visualization of the optimization process, typically showing the objective function, the sampled points, the predicted mean and uncertainty of the Gaussian Process model, and the acquisition function over the iterations. This helps understand how the algorithm explored the space and converged towards the optimum. The calculated result as follows,
<img width="1139" alt="image" src="https://github.com/user-attachments/assets/bbd08fb8-85c3-4dd7-a48d-0988a9fe2803" />


# Active learning with Gaussian processes
Function definition as before.

    bounds_1d = [0.0 1.0]
    n_al_iterations = 50
    n_initial_al = 3
    seed_al = 5678
Same as the initial definition.
    
    n_test_points = 200
    Random.seed!(123)
    X_test_mat = BayesianOpt.Utils.sample_randomly(bounds_1d, n_test_points)
    Y_test_vec = [complex_1d_func_noiseless(X_test_mat[:, i]) for i in 1:n_test_points]
This section prepares a separate set of test data points to evaluate the performance of the active learning process.

`n_test_points`: Defines the number of points in the test set.

`Random.seed!(123)`: Sets a seed for the random number generator specifically for generating the test points, ensuring this test set is reproducible.

`X_test_mat`: Generates `n_test_points` random input values within the specified `bounds_1d`.

`Y_test_vec`: Evaluates a noiseless version of the objective function (`complex_1d_func_noiseless`, assumed to be defined elsewhere) at each of the test points. This provides a clean baseline to measure how well the active learning model is approximating the true function.

Running Active Learning

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
This is the main function call that executes the Active Learning process. Unlike basic Bayesian Optimization, Active Learning often focuses on building an accurate model of the function, and this setup includes test data for evaluation.

`complex_1d_func`: The objective function to be learned/optimized.

`bounds_1d`: The search space.

`X_test=X_test_mat` and `Y_test=Y_test_vec`: Provides the generated test data for evaluating the model's performance during the learning process.

`n_al_iterations`: The number of active learning iterations.

`n_initial_points=n_initial_al`: Specifies the number of initial random points.

`variance_maximizer=LBFGS`: Sets the method used to select the next point to sample. In active learning, this often involves maximizing uncertainty or variance to improve the model's knowledge in less explored areas. L-BFGS is used here.

`verbose=true`: Enables detailed output during the process.

`random_seed=seed_al`: Sets the random seed for the active learning run.

`optimize_gp_hypers=true`: Instructs the algorithm to optimize the hyperparameters of the internal Gaussian Process model at each iteration.

Viewing Results
    
    plot_results(al_results)
The plotting function is the same as above. The plotting results are different from Bayesian optimization because a test set has been added here, showing the error of the test results.
<img width="921" alt="image" src="https://github.com/user-attachments/assets/a30c345e-29d0-4504-87aa-684fc4191c22" />



# Active learning with random forests
The boundary is defined as  
  
    bounds_1d = reshape([0.0, 2.0], 1, 2) 
We need to set the parameters for the random forest. See DecisionTree.jl for a detailed description of the parameters.

    rf_params = Dict(
        :max_depth => -1,           
        :min_samples_leaf => 3,    
        :min_samples_split => 2,   
        :partial_sampling => 0.7   
        # :n_subfeatures => -1     
    )
Test set configuration:

    X_test_mat = BayesianOpt.Utils.sample_randomly(bounds_1d, n_test_points)
    Y_test_vec = [complex_1d_func_noiseless(X_test_mat[:, i]) for i in 1:n_test_points]

The function for active learning uses `active_learn_rf`

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
Drawing method is the same as described above.

    plot_results(al_rf_results)

<img width="911" alt="image" src="https://github.com/user-attachments/assets/b40a02d2-e3f0-446c-ac73-3a7fe4df5a18" />

# Active learning with mix GP and RF, including the prior for GP noise

Objective Functions

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
These functions define the target we are working with. `complex_1d_func_noisy` includes a small amount of random noise (`randn()*0.03`) to simulate real-world uncertainty, and is used for sampling points during the active learning process. `complex_1d_func_noiseless` is the true underlying function without noise, used here to generate a clean test set for evaluating the model's performance. Both are one-dimensional functions (`x_vec[1]`) with multiple features.
    
    bounds_1d = [0.0 1.0]
    n_al_iterations = 100
    n_initial_al = 10
    seed_al = 5678

These parameter settings are the same as before.

Test set configuration

    n_test_points = 200
    Random.seed!(123)
    X_test_mat = BayesianOpt.Utils.sample_randomly(bounds_1d, n_test_points)
    Y_test_vec = [complex_1d_func_noiseless(X_test_mat[:, i]) for i in 1:n_test_points]

Random Forest Parameters (Mix Model)

    rf_n_trees_hybrid = 2000
    rf_params_hybrid = Dict(
        :min_samples_leaf => 3,
        :max_depth => -1
    )

Gaussian Process Parameters:
    
    kernel_gp_res = SEIso(0.0, 0.0)
    noise_prior = Normal(-6.0, 1.5)

Maximizer Configuration

    maximizer_choice = LBFGS
    maximizer_config = Dict(:lbfgs_starts => 15)
These parameters configure the optimization method used to find the next point to sample during active learning. This point is typically chosen by maximizing an acquisition function (like variance).

`maximizer_choice`: Specifies the optimization algorithm to use. `LBFGS` is a quasi-Newton method.

`maximizer_config`: A dictionary providing configuration options for the chosen maximizer, such as the number of random starting points for L-BFGS to help avoid getting stuck in local optima.

Running Hybrid RF+GP Active Learning

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
    
`complex_1d_func_noisy`: The objective function used to obtain noisy observations.

`bounds_1d`: The search space.

`n_al_iterations`: The number of active learning iterations.

`n_initial_points=n_initial_al`: Specifies the number of initial random points.

`X_test=X_test_mat` and `Y_test=Y_test_vec`: Provides the generated test data for evaluating the model's performance during the learning process.

`rf_n_trees` and `rf_kwargs`: Pass the Random Forest configuration.

`kernel` and `logNoise_prior`: Pass the Gaussian Process kernel and noise prior configuration.

`optimize_gp_hypers=true`: Instructs the algorithm to optimize the hyperparameters of the internal Gaussian Process model at each iteration.

`variance_maximizer` and `maximizer_kwargs`: Pass the configuration for the method used to select the next point (maximizing variance in this case).

`verbose=true`: Enables detailed output during the process.

`random_seed=seed_al`: Sets the random seed for reproducibility.

Drawing is the same as above

    results_plot = plot_results(al_results_hybrid,
                            objective_func = complex_1d_func_noiseless,
                            plot_model_fit=true,
                            plot_parity_train=true,
                            plot_parity_test=true,
                            plot_test_performance=true
                           )

<img width="918" alt="image" src="https://github.com/user-attachments/assets/c51cb455-c219-4a50-b7c7-469ea840fbd5" />


The calculation methods for higher dimensions are similar to those here and can be found in the `example` folder.
