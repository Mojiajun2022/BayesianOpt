module RandomForestModel

using DecisionTree
using Statistics
using LinearAlgebra

export train_rf, predict_rf_mean_variance, RandomForestRegressor

const RandomForestRegressor = DecisionTree.RandomForestRegressor

function train_rf(X_train::AbstractMatrix, Y_train::AbstractVector;
    n_trees::Int=100, kwargs...)
    if isempty(Y_train)
        error("Cannot train RF with empty data.")
    end
    X_train_transposed = Matrix(X_train')
    Y_train_concrete = Vector{Float64}(Y_train)
    model = RandomForestRegressor(n_trees=n_trees; kwargs...)
    DecisionTree.fit!(model, X_train_transposed, Y_train_concrete)
    return model
end


"""
    predict_rf_mean_variance(model::RandomForestRegressor, X_pred::AbstractMatrix)

Predicts the mean and variance of predictions from individual trees in the forest.
Uses manual tree iteration as a workaround for apply_forest dispatch issues.
"""
function predict_rf_mean_variance(model::RandomForestRegressor, X_pred::AbstractMatrix)
    if size(X_pred, 2) == 0
        return Float64[], Float64[]
    end
    X_pred_transposed = Matrix(X_pred')
    n_samples = size(X_pred_transposed, 1)

    local trees

    if hasfield(typeof(model), :ensemble) &&
       model.ensemble !== nothing &&
       hasfield(typeof(model.ensemble), :trees) &&
       model.ensemble.trees !== nothing &&
       !isempty(model.ensemble.trees)
        trees = model.ensemble.trees
    else
        println("Debug Info: model type=", typeof(model), ", has :ensemble=", hasfield(typeof(model), :ensemble))
        if hasfield(typeof(model), :ensemble) && model.ensemble !== nothing
            println("Debug Info: model.ensemble type=", typeof(model.ensemble), ", has :trees=", hasfield(typeof(model.ensemble), :trees))
            if hasfield(typeof(model.ensemble), :trees) && model.ensemble.trees !== nothing
                println("Debug Info: isempty(model.ensemble.trees)=", isempty(model.ensemble.trees))
            end
        end
        error("Cannot access individual trees. Expected 'model.ensemble.trees' structure not found, or the tree collection is empty.")
    end

    n_trees = length(trees)
    if n_trees == 0
        error("RF model's tree collection is empty after access.")
    end
    tree_predictions = Matrix{Float64}(undef, n_samples, n_trees)

    for i in 1:n_trees
        tree = trees[i]
        try
            tree_predictions[:, i] = apply_tree(tree, X_pred_transposed)
        catch e
            println("Error applying tree $i: $e")
            tree_predictions[:, i] .= NaN
        end
    end

    valid_tree_cols = findall(j -> !any(isnan, @view(tree_predictions[:, j])), 1:n_trees)
    if isempty(valid_tree_cols)
        println("Warning: All trees failed prediction.")
        return fill(NaN, n_samples), fill(NaN, n_samples)
    elseif length(valid_tree_cols) < n_trees
        println("Warning: $(n_trees - length(valid_tree_cols)) trees failed prediction.")
    end

    valid_predictions = @view tree_predictions[:, valid_tree_cols]

    predicted_means = vec(mean(valid_predictions, dims=2))
    predicted_variances = if length(valid_tree_cols) > 1
        vec(var(valid_predictions, dims=2; corrected=false))
    else
        zeros(Float64, n_samples)
    end

    predicted_variances = max.(predicted_variances, 0.0)
    predicted_means[isnan.(predicted_means)] .= 0.0
    predicted_variances[isnan.(predicted_variances)] .= 0.0

    return predicted_means, predicted_variances

end

end