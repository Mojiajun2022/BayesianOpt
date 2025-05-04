module BayesianOpt

include("utils.jl")
include("gaussian_process.jl")
include("random_forest_model.jl")
include("acquisition_functions.jl")
include("maximizers.jl")
include("optimization_loop.jl")
include("visualization.jl")

using .GPModel: GPE, predict_with_gp
using .RandomForestModel: RandomForestRegressor
using .AcquisitionFunctions: AcquisitionType, EI, PI, UCB
using .Maximizers: MaximizerType, GRID, RANDOM, LBFGS, PROBSAMPLE
using .OptimizationLoop: bayes_optimize, active_learn, active_learn_rf, active_learn_residual_rf_gp
using .Visualization: plot_bo_results, plot_model_parity, plot_al_test_performance, plot_test_set_parity, plot_rf_parity, plot_rf_1d_results, plot_results


export bayes_optimize
export AcquisitionType, EI, PI, UCB
export MaximizerType, GRID, RANDOM, LBFGS, PROBSAMPLE

export active_learn
export active_learn_rf
export active_learn_residual_rf_gp
export plot_bo_results
export plot_model_parity
export plot_test_set_parity
export plot_al_test_performance
export plot_rf_parity
export plot_rf_1d_results
export predict_with_gp
export plot_results
# export GPE
# export RandomForestRegressor

end