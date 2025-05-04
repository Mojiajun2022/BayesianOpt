module AcquisitionFunctions

using Distributions
using LinearAlgebra
using GaussianProcesses

export calculate_acquisition, AcquisitionType, EI, PI, UCB

@enum AcquisitionType EI PI UCB

const ε_std = 1e-9


function expected_improvement(μ::Real, σ::Real, y_best::Real, maximize::Bool; ξ::Real=0.01)
    if σ < ε_std
        return 0.0
    end
    improvement_sign = maximize ? 1.0 : -1.0
    diff = improvement_sign * (μ - y_best) - ξ
    Z = diff / σ
    ei = diff * cdf(Normal(), Z) + σ * pdf(Normal(), Z)
    return max(0.0, ei)
end

function probability_of_improvement(μ::Real, σ::Real, y_best::Real, maximize::Bool; ξ::Real=0.01)
    if σ < ε_std
        improvement_sign = maximize ? 1.0 : -1.0
        diff = improvement_sign * (μ - y_best) - ξ
        return diff > 0 ? 1.0 : 0.0
    end
    improvement_sign = maximize ? 1.0 : -1.0
    diff = improvement_sign * (μ - y_best) - ξ
    Z = diff / σ
    return cdf(Normal(), Z)
end

function upper_confidence_bound(μ::Real, σ::Real, maximize::Bool; κ::Real=2.0)
    return μ + κ * σ
end

"""
    calculate_acquisition(gp::GPE, X_pred::AbstractMatrix, acq_type::AcquisitionType, maximize::Bool, y_best::Real; ξ::Real=0.01, κ::Real=2.0)

Calculate the acquisition function values on the given point set X_pred.
Requires gp model for prediction.
"""
function calculate_acquisition(gp::GPE,
    X_pred::AbstractMatrix,
    acq_type::AcquisitionType,
    maximize::Bool,
    y_best::Real;
    ξ::Real=0.01,
    κ::Real=2.0)

    if size(X_pred, 2) == 0
        return Float64[]
    end
    μ_pred, σ²_pred = predict_y(gp, X_pred)
    σ_pred_safe = sqrt.(max.(σ²_pred, 1e-12))

    n_points = size(X_pred, 2)
    acq_values = Vector{Float64}(undef, n_points)

    for i in 1:n_points
        μ = μ_pred[i]
        σ = σ_pred_safe[i]

        if acq_type == EI
            acq_values[i] = expected_improvement(μ, σ, y_best, maximize, ξ=ξ)
        elseif acq_type == PI
            acq_values[i] = probability_of_improvement(μ, σ, y_best, maximize, ξ=ξ)
        elseif acq_type == UCB
            acq_values[i] = upper_confidence_bound(μ, σ, maximize, κ=κ)
        else
            error("Unknown acquisition function type: $acq_type")
        end
    end
    return acq_values
end

end