module GPModel

using GaussianProcesses
using Distributions
using LinearAlgebra
using Random


export update_gp!, predict_with_gp, create_initial_gp, GPE

"""
Creates an initial or updates an existing Gaussian process model.
Allows for specifying a prior for logNoise. Kernel priors should be pre-set on the passed-in kernel object.
"""
function update_gp!(gp::Union{GPE,Nothing},
    X::AbstractMatrix,
    Y::AbstractVector;
    kernel=SEIso(0.0, 0.0),
    logNoise=-2.0,
    mean_func=MeanZero(),
    optimize_hypers=true,
    logNoise_prior::Union{Distribution,Nothing}=nothing
)

    if isempty(X) || isempty(Y)
        println("Warning: Training data is empty, unable to create or update GP model.")
        return nothing
    end
    current_dim = size(X, 1)
    if gp === nothing || gp.dim != current_dim
        new_gp = GPE(X, Y, mean_func, kernel, logNoise)
        println("Created new GP. Kernel: ", new_gp.kernel, " Initial logNoise: ", new_gp.logNoise)
    else
        new_gp = GPE(X, Y, gp.mean, gp.kernel, gp.logNoise)
        if gp.logNoise != logNoise
            new_gp.logNoise = logNoise
        end
        println("Updated existing GP. Kernel: ", new_gp.kernel, " Current logNoise: ", new_gp.logNoise)
    end

    if logNoise_prior !== nothing
        try
            set_priors!(new_gp, Dict(:logNoise => logNoise_prior))
            println("Successfully set logNoise prior: ", logNoise_prior)
        catch e
        end
    else
        println("No logNoise prior provided or set.")
    end


    if optimize_hypers
        println("Optimizing GP hyperparameters (considering the set priors)...")
        try
            optimize!(new_gp, domean=false, kern=true, noise=true)
            println("GP hyperparameter optimization complete.")
            println("  Optimized Kernel: ", new_gp.kernel)
            logNoise_value = new_gp.logNoise.value
            println("  Optimized logNoise: ", round(logNoise_value, digits=3))
        catch e
            println("GP hyperparameter optimization failed: $e. Using previous or default hyperparameters.")
        end
    end
    return new_gp
end


"""
Predict using a Gaussian process model.
Returns the predicted mean and standard deviation.
"""
function predict_with_gp(gp::GPE, X_pred::AbstractMatrix)
    if size(X_pred, 1) != gp.dim
        error("The dimension of the prediction points ($(size(X_pred,1))) does not match the dimension of the GP model ($(gp.dim).")
    end
    if size(X_pred, 2) == 0
        return Float64[], Float64[]
    end
    μ, σ² = predict_y(gp, X_pred)
    σ = sqrt.(max.(σ², 1e-12))
    return μ, σ
end

"""
Create initial GP model (optional, but not really needed now because `update_gp!` can handle `gp=nothing`)
"""
function create_initial_gp(X_init::AbstractMatrix, Y_init::AbstractVector;
    kernel=SEIso(0.0, 0.0),
    logNoise=-2.0,
    mean_func=MeanZero(),
    logNoise_prior=nothing)
    if isempty(X_init)
        println("Cannot create GP because empty initial data is provided.")
        return nothing
    end
    return update_gp!(nothing, X_init, Y_init;
        kernel=kernel, logNoise=logNoise, mean_func=mean_func,
        optimize_hypers=true, logNoise_prior=logNoise_prior)
end


end