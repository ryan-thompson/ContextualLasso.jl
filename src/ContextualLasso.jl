# =================================================================================================#
# Description: Implementation of the contextual lasso
# Author: Ryan Thompson
# =================================================================================================#

module ContextualLasso

import CUDA, Flux, Gadfly, Printf, Statistics, Zygote

export classo, coef, predict, plot

#==================================================================================================#
# Function that reimplements Flux.early_stopping with <= instead of <
#==================================================================================================#

function early_stopping(f, delay; distance = -, init_score = 0, min_dist = 0)
    trigger = let best_score = init_score
      (args...; kwargs...) -> begin
        score = f(args...; kwargs...)
        Δ = distance(best_score, score)
        best_score = Δ < 0 ? best_score : score
        return Δ <= min_dist
      end
    end
    return Flux.patience(trigger, delay)
  end

#==================================================================================================#
# Function to (group-wise) soft-threshold a matrix or vector
#==================================================================================================#

function threshold(beta, theta, group_info)

    if isnothing(group_info)

        # If no grouping, directly threshold individual betas 
        Flux.softshrink(beta, theta)

    else

        group, groups = group_info

        # If grouping, compute group-wise norms ||z||_2
        group_norm = vcat(map(k -> sqrt.(sum(beta[k .== group, :] .^ 2, dims = 1)), groups)...)

        # Threshold group-wise norms and rescale S(||βₖ||,λ) / ||βₖ||_2
        group_norm_thresh = Flux.softshrink(group_norm, theta) ./ 
            (group_norm .+ eps(eltype(group_norm)))

        # Multiply above by βₖ
        vcat(map(k -> beta[k .== group, :] .* group_norm_thresh[k .== groups, :], groups)...)

    end

end

#==================================================================================================#
# Function to (group-wise) project a matrix or vector onto l1-ball
#==================================================================================================#

function project(beta, lambda, group_info)

    if isnothing(group_info)

        # If no grouping, directly project individual betas 
        project_l1(beta, lambda = lambda)

    else
        
        group, groups = group_info

        # If grouping, compute group-wise norms ||z||_2
        group_norm = vcat(map(k -> sqrt.(sum(beta[k .== group, :] .^ 2, dims = 1)), groups)...)

        # Threshold group-wise norms
        group_norm_thresh, theta = project_l1(group_norm, lambda = lambda)
        
        # βₖ * P(||βₖ||,λ) / ||βₖ||_2
        vcat(map(k -> beta[k .== group, :] .* group_norm_thresh[k .== groups, :] ./ 
            (group_norm[k .== groups, :] .+ eps(eltype(group_norm))), groups)...), theta

    end

end

function project_l1(beta; lambda = lambda)

    # Check which device beta exists on
    if isa(beta, Matrix)
        device = Flux.cpu
    else
        device = Flux.gpu
    end

    # Save input dims
    dims = size(beta)
    nlambda = dims[2] * lambda

    # If already in l1 ball or lambda=0 then exit early
    if sum(abs.(beta)) <= nlambda
        return beta, 0
    elseif isapprox(lambda, zero(lambda), atol = sqrt(eps(lambda)))
        return zero(beta), Inf
    end

    # Flatten to vector
    beta = vec(beta)

    # Remove signs
    beta_abs = abs.(beta)

    # Run algorithm for projection onto simplex by Duchi et al. (2008, ICML)
    # beta_sort = sort(beta_abs, rev = true) # Does not work on GPU
    ind = sortperm(beta_abs, rev = true)
    beta_sort = beta_abs[ind]
    # beta_sort = reverse(beta_abs[sortperm(beta_abs)]) # Treats ties differently to above;...
    # ...can lead to different results on GPU because ties occur more often with fp32
    csum = cumsum(beta_sort)
    indices = device(collect(1:prod(dims)))
    max_j = maximum((beta_sort .* indices .> csum .- nlambda) .* indices)
    # lm = max_j * (csum[max_j] - nlambda) / max_j # Does not work on GPU
    theta = (sum(beta_sort[1:max_j]) - nlambda) / max_j

    # Threshold beta
    beta_abs = max.(beta_abs .- theta, 0)
    beta = beta_abs .* sign.(beta)

    # Return to original shape
    reshape(beta, dims), theta

end

# Below gradient is adapted from https://arxiv.org/abs/2310.15627
function Zygote.rrule(::typeof(project_l1), b̃; lambda)

    # Compute optimal solution
    b̂, theta = project_l1(b̃, lambda = lambda)

    # Configure vector-Jacobian product (vJp)
    function b̂_pullback(db̂_tuple)
        db̂ = db̂_tuple[1]
        dims = size(db̂)
        db̂ = vec(db̂)
        b̂ = vec(b̂)
        if theta == 0
            db̃ = db̂
        elseif theta == Inf
            db̃ = zero(db̂)
        else
            A = b̂ .≠ 0
            alpha =  1 / sum(A) * sum(sign.(b̂) .* db̂)
            db̃ =  A .* db̂ .- alpha .* sign.(b̂)
        end
        db̃ = reshape(db̃, dims)
        (Zygote.NoTangent(), db̃, Zygote.NoTangent())
    end

    (b̂, theta), b̂_pullback

end

#==================================================================================================#
# Function to project a matrix or vector onto closest signed matrix or vector
#==================================================================================================#

function project_sign(beta, s)
    s .* Flux.relu(s .* beta) + iszero.(s) .* beta
end

#==================================================================================================#
# Function to generate neural network architecture
#==================================================================================================#

function gennet(hidden_layers, sign_constraint, dropout, p, m, activation_fun)

    # Determine number of layers
    if dropout == 0
        nlayer = length(hidden_layers) + 1
    else
        nlayer = length(hidden_layers) * 2 + 1
        
    end

    # Build layers
    layer = Vector{Union{Flux.Dense, Flux.Dropout}}(undef, nlayer)
    j = 0
    for i in 1:nlayer
        j += 1
        if i == 1 == nlayer
            layer[i] = Flux.Dense(m, p)
        elseif i == 1
            layer[i] = Flux.Dense(m, hidden_layers[j], activation_fun)
        elseif i == nlayer
            layer[i] = Flux.Dense(hidden_layers[j - 1], p)
        elseif dropout == false || mod(i, 2) == 1
            layer[i] = Flux.Dense(hidden_layers[j - 1], hidden_layers[j], activation_fun)
        else
            layer[i] = Flux.Dropout(dropout)
            j -= 1
        end
    end

    # Unroll layers into chain
    if iszero(sign_constraint)
        Flux.Chain(layer...)
    else
        Flux.Chain(layer..., x -> project_sign(x, sign_constraint))
    end


end

#==================================================================================================#
# Function to train model
#==================================================================================================#

function train(model, lambda, x, z, y, x_val, z_val, y_val, loss, intercept, group_info, optimiser, 
    epoch_max, early_stop, patience, verbose, verbose_freq)

    # Save data dimensions
    n_val = size(x_val, 2)

    # Instantiate optimiser and model parameters
    optim = optimiser()
    optim_state = Flux.setup(optim, model)

    theta = Ref(0.0)

    # Create objective function
    function objective(model; x = x, z = z, y = y, par = lambda, group_info = group_info, 
            agg = Statistics.mean, inference = false)
        beta = model(z)
        beta_int = beta[1 + intercept:end, :]
        if inference
            beta_int = threshold(beta_int, par, group_info)
        else
            beta_int, theta[] = project(beta_int, par, group_info)
        end
        y_hat = sum(x .* beta_int, dims = 1)
        if intercept
            y_hat += transpose(beta[1, :])
        end
        loss(y_hat, y, agg = agg)
    end

    # Initialise variables
    train_loss = objective(model)
    val_loss = objective(model, x = x_val, z = z_val, y = y_val, par = theta[], 
        group_info = group_info, inference = true)
    epochs = 0

    # Set convergence criterion
    if early_stop
        model_best = deepcopy(model)
        theta_best = theta[]
        train_loss_best = train_loss
        val_loss_best = val_loss
        epochs_best = 0
        converge = early_stopping(x -> x, patience, init_score = val_loss_best)
    else
        converge = early_stopping(x -> x, patience, init_score = Inf)
    end

    # Run optimisation
    for epoch in 1:epoch_max

        # Record training loss and gradients
        train_loss, grad = Flux.withgradient(objective, model)

        # Record validation loss
        val_loss = objective(model, x = x_val, z = z_val, y = y_val, par = theta[], 
            group_info = group_info, inference = true)
        
        # Print status update
        if verbose && epoch % verbose_freq == 0
            Printf.@printf("\33[2K\rEpoch: %i, Train loss: %.4f, Valid loss: %.4f", epoch, 
                train_loss, val_loss)
        end

        # Check for improvement
        if early_stop && val_loss < val_loss_best
            model_best = deepcopy(model)
            theta_best = theta[]
            train_loss_best = train_loss
            val_loss_best = val_loss
            epochs_best = epochs
        end

        # Check for convergence
        if early_stop 
            converge(val_loss) && break
        else
            converge(train_loss) && break
        end

        # Update parameters
        if epoch < epoch_max
            Flux.update!(optim_state, model, grad[1])
            epochs = epoch
        end

    end

    # Update model to that which achieved lowest validation loss if using early stopping
    if early_stop
        model = model_best
        theta = theta_best
        train_loss = train_loss_best
        val_loss = val_loss_best
        epochs = epochs_best
    else
        theta = theta[]
    end

    # Compute validation standard errors and sparsity levels for plotting
    val_loss_se = objective(model; x = x_val, z = z_val, y = y_val, par = theta, 
        group_info = group_info, agg = x -> Statistics.std(x, corrected = false), 
        inference = true) / sqrt(n_val)
    val_nonzero = sum(threshold(model(z_val)[1 + intercept:end, :], theta, group_info) .!= 0) / 
        n_val

    model, theta, train_loss, val_loss, val_loss_se, val_nonzero, epochs

end

#==================================================================================================#
# Function to train polished model
#==================================================================================================#

function train_polish(model, mask, mask_val, x, z, y, x_val, z_val, y_val, loss, intercept, 
    group_info, optimiser, epoch_max, early_stop, patience, verbose, verbose_freq)

    # Instantiate optimiser and model parameters
    optim = optimiser()
    optim_state = Flux.setup(optim, model)
    
    # Create objective function
    function objective(model, x, z, y, mask, group_info)
        beta = model(z)
        beta_int = beta[1 + intercept:end, :]
        beta_int = beta_int .* mask
        y_hat = sum(x .* beta_int, dims = 1)
        if intercept
            y_hat += transpose(beta[1, :])
        end
        loss(y_hat, y)
    end
    
    # Initialise variables
    train_loss = objective(model, x, z, y, mask, group_info)
    val_loss = objective(model, x_val, z_val, y_val, mask_val, group_info)
    epochs = 0

    # Set convergence criterion
    if early_stop
        model_best = deepcopy(model)
        train_loss_best = train_loss
        val_loss_best = val_loss
        epochs_best = 0
        converge = early_stopping(x -> x, patience, init_score = val_loss_best)
    else
        converge = early_stopping(x -> x, patience, init_score = Inf)
    end

    # Run optimisation
    for epoch in 1:epoch_max

        # Record validation loss
        val_loss = objective(model, x_val, z_val, y_val, mask_val, group_info)

        # Record training loss and gradients
        train_loss, grad = Flux.withgradient(objective, model, x, z, y, mask, group_info)

        # Print status update
        if verbose && epoch % verbose_freq == 0
            Printf.@printf("\33[2K\rRelaxed fit - Epoch: %i, Train loss: %.4f, Valid loss: %.4f", 
                epoch, train_loss, val_loss)
        end

        # Check for improvement
        if early_stop && val_loss < val_loss_best
            model_best = deepcopy(model)
            train_loss_best = train_loss
            val_loss_best = val_loss
            epochs_best = epochs
        end

        # Check for convergence
        if early_stop 
            converge(val_loss) && break
        else
            converge(train_loss) && break
        end

        # Update parameters
        if epoch < epoch_max
            Flux.update!(optim_state, model, grad[1])
            epochs = epoch
        end

    end

    # Update model to that which achieved lowest validation loss if using early stopping
    if early_stop
        model = model_best
        train_loss = train_loss_best
        val_loss = val_loss_best
        epochs = epochs_best
    end

    model, train_loss, val_loss, epochs

end

#==================================================================================================#
# Type for contextual lasso
#==================================================================================================#

struct ContextualLassoFit
    model::Vector{Flux.Chain} # Sequence of fitted models
    model_polish::Vector{Flux.Chain} # Sequence of fitted models polished on active set
    lambda::Vector{Float32} # Sequence of regularisation parameters
    gamma::Vector{Float32} # Sequence of relaxation parameters
    intercept::Bool # Whether an intercept was fit
    group_info::Union{Nothing, Tuple} # Grouping of the explanatory features
    relax::Bool # Whether a relaxed fit was performed
    lambda_min::Float32 # Best regularization parameter
    gamma_min::Float32 # Best relaxation parameter 
    lambda_1se::Float32 # Regularization parameter within one standard error of best
    gamma_1se::Float32 # Relaxation parameter within one standard error of best
    theta::Vector{Float32} # Estimated thresholding parameter
    x_mean::Matrix{Float32} # Mean of explanatory features over training set
    x_sd::Matrix{Float32} # Standard deviation of explanatory features over training set
    z_mean::Matrix{Float32} # Mean of contextual features over training set
    z_sd::Matrix{Float32} # Standard deviation of contextual features over training set
    y_mean::Float32 # Mean of response over training set
    y_sd::Float32 # Standard deviation of response over training set
    train_loss::Vector{Float32} # Sequence of mean training losses
    val_loss::Vector{Float32} # Sequence of mean validation losses
    val_loss_se::Vector{Float32} # Sequence of standard error of mean validation losses
    val_nonzero::Vector{Float32} # Sequence of number of nonzeros over validation set
    epochs::Vector{Int} # Sequnce of number of epochs
    train_loss_polish::Vector{Float32} # Sequence of mean training losses from polished models
    val_loss_polish::Vector{Float32} # Sequence of mean validation losses from polished models
    epochs_polish::Vector{Int} # Sequence of number of epochs for polished models
    val_loss_relax::Matrix{Float32} # Sequence of mean validation losses from polished models
    val_loss_se_relax::Matrix{Float32} # Sequence of standard error of mean validation losses 
    # from polished models
end

#==================================================================================================#
# Function to perform a contextual lasso fit
#==================================================================================================#

"""
    classo(x, z, y, x_val, z_val, y_val; <keyword arguments>)

Performs a contextual lasso fit to explantory features `x`, contextual features `z`, and response \
`y`. The training data are `x`, `z`, and `y`, and the validation data are `x_val`, `z_val`, and \
`y_val`.

# Arguments
- `loss = Flux.mse`: a loss function of the form `loss(y_hat, y)`; must accept linear predictors \
y_hat = x * beta(z) as predictions; for regression use `Flux.mse` and for classification use \
`Flux.logitbinarycrossentropy`.
- `intercept = true`: whether to include an intercept in the linear model.
- `group = nothing`: an optional grouping of the explanatory features; if provided, should be a \
vector of length `size(x, 2)` with the jth element identifying the group that the jth explanatory \
feature belongs to.
- `lambda = nothing`: an optional sequence of regularisation parameters; if empty will be computed \
as an equispaced spaced sequence of length `lambda_n` from `lambda_max` to zero, where \
`lambda_max` is automatically computed from the data to ensure all explanatory features are \
included.
- `lambda_n = 50`: the number of regularisation parameters to evaluate.
- `relax = false`: whether to perform a relaxed fit; if `true` the models are refit on their \
active sets without any regularisation.
- `gamma = collect(range(0, 1, 11))`: a sequence of relaxation paramters; `gamma = 0` yields \
the original fit with full shrinkage and `gamma = 1` yields the fully relaxed fit with no shrinkage.
- `device = Flux.gpu`: the device to train on; either `Flux.cpu` for CPU or `Flux.gpu` for GPU.
- `optimiser = Flux.Adam`: an optimiser from Flux to use for training.
- `epoch_max = 10000`: the maximum number of training epochs.
- `early_stop = true`: whether to use early stopping; if `true` convergence is monitored on the \
validation set or if `false` convergence is monitored on the training set.
- `patience = 30`: the number of epochs to wait before declaring convergence.
- `hidden_layers = [128, 128]`: the configuration of the feedforward neural network; by default \
produces a network with two hidden dense layers of 128 neurons each.
- `dropout = 0`: the dropout rate; a dropout layer is included after each dense layer; if \
`0` no dropout layers are included.
- `initialise = "warm"`: how to initialise the optimiser; `"warm"` to warm start the optimiser \
using the previous solution along the regularisation path or `"cold"` to cold start the optimiser \
with a random initialisation.
- `sign_constraint = fill(0, size(x, 2) + intercept)`: a vector of length `size(x, 2) + intercept` \
with elements 1, -1, or 0 to constrain the signs of coefficients; 1 indicates only a positive sign \
is allowed, -1 indicates only a negative negative is allowed, and 0 indicates a positive or \
negative sign is allowed.
- `verbose = true`: whether to print status updates during training.
- `verbose_freq = 10`: the number of epochs to wait between status updates.
- `standardise_x = true`: whether to standardise the explanatory features to have zero mean and \
unit variance; ensures regularisation is equitable and helps during training.
- `standardise_z = true`: whether to standardise the contextual features to have zero mean and \
unit variance; helps during training.
- `standardise_y = true`: whether to standardise the response to have zero mean and unit variance; \
helps during training; `false` unless `loss == Flux.mse`.
- `activation_fun = Flux.relu`: an activation function to use in the hidden layers.
``

See also [`coef`](@ref), [`predict`](@ref), [`plot`](@ref).
"""
function classo(x::Matrix{<:Real}, z::Matrix{<:Real}, y::Vector{<:Real}, x_val::Matrix{<:Real}, 
    z_val::Matrix{<:Real}, y_val::Vector{<:Real}; loss::Function = Flux.mse, intercept::Bool = true, 
    group::Union{Vector{<:Int}, Nothing} = nothing, 
    lambda::Union{Real, Vector{<:Real}, Nothing} = nothing, lambda_n::Int = 50, relax::Bool = false, 
    gamma::Union{Real, Vector{<:Real}} = collect(range(0, 1, 11)), device::Function = Flux.gpu, 
    optimiser::DataType = Flux.Adam, epoch_max::Int = 10000, early_stop::Bool = true, 
    patience::Int = 30, hidden_layers::Vector{<:Any} = [128, 128], dropout::Real = 0, 
    initialise::String = "warm", sign_constraint::Vector{<:Int} = fill(0, size(x, 2) + intercept), 
    verbose::Bool = true, verbose_freq::Int = 10, standardise_x::Bool = true, 
    standardise_z::Bool = true, standardise_y::Bool = true, activation_fun = Flux.relu)

    # Validate arguments
    initialise in ["warm", "cold"] || error("""initialise should be "warm" or "cold".""")

    # Save data dimensions
    n, p = size(x)
    m = size(z, 2)
    n_val = size(x_val, 1)

    # Set sign constraint vector to zero if not provided (i.e., no sign constraint)
    if isnothing(sign_constraint)
        sign_constraint = zeros(p)
    end

    if isnothing(group)
        group_info = nothing
    else
        group_info = (group, unique(group))
    end

    # Flux defaults to f32 model parameters
    x, z, y = Flux.f32(x), Flux.f32(z), Flux.f32(y)
    x_val, z_val, y_val = Flux.f32(x_val), Flux.f32(z_val), Flux.f32(y_val)

    # Standardise explanatory features
    if standardise_x && intercept
        x_mean = mapslices(Statistics.mean, x, dims = 1)
        x_sd = mapslices(x -> Statistics.std(x, corrected = false), x, dims = 1)
    elseif standardise_x && !intercept
        x_mean = zeros(1, p)
        x_sd = mapslices(x -> Statistics.std(x, corrected = false, mean = 0), x, dims = 1)
    else
        x_mean = zeros(1, p)
        x_sd = ones(1, p)
    end
    if any(x_sd .== 0)
        x_sd[x_sd .== 0] .= 1 # Handles constants
    end
    x = (x .- x_mean) ./ x_sd
    x_val = (x_val .- x_mean) ./ x_sd

    # Standardise contextual features
    if standardise_z
        z_mean = mapslices(Statistics.mean, z, dims = 1)
        z_sd = mapslices(z -> Statistics.std(z, corrected = false), z, dims = 1)
    else
        z_mean = zeros(1, m)
        z_sd = ones(1, m)
    end
    if any(z_sd .== 0)
        z_sd[z_sd .== 0] .= 1 # Handles constants
    end
    z = (z .- z_mean) ./ z_sd
    z_val = (z_val .- z_mean) ./ z_sd

    # Standardise response
    if standardise_y && intercept && loss == Flux.mse
        y_mean = Statistics.mean(y)
        y_sd = Statistics.std(y, corrected = false)
    elseif standardise_y && !intercept && loss == Flux.mse
        y_mean = 0
        y_sd = Statistics.std(y, corrected = false, mean = 0)
    else
        y_mean = 0
        y_sd = 1
    end
    if y_sd == 0
        y_sd = 1 # Handles constants
    end
    y = (y .- y_mean) ./ y_sd
    y_val = (y_val .- y_mean) ./ y_sd

    # Transpose because Flux expects features in rows and samples in columns
    x, z, y = transpose(x), transpose(z), transpose(y)
    x_val, z_val, y_val = transpose(x_val), transpose(z_val), transpose(y_val)

    # Move data to correct device
    x, z, y = device(x), device(z), device(y)
    x_val, z_val, y_val = device(x_val), device(z_val), device(y_val)

    # Compute lambda sequence
    if !isnothing(lambda)
        lambda_n = length(lambda)
        if !isa(lambda, Vector)
            lambda = [lambda]
        end
        lambda = Float32.(lambda)
    end
    if !isa(gamma, Vector)
        gamma = [gamma]
    end
    gamma_n = length(gamma)

    # Allocate space for models, objectives, and losses
    model = Vector{Flux.Chain}(undef, lambda_n)
    model_polish = Vector{Flux.Chain}(undef, lambda_n)
    theta = Vector{Float32}(undef, lambda_n)
    train_loss = Vector{Float32}(undef, lambda_n)
    val_loss = Vector{Float32}(undef, lambda_n)
    val_loss_se = Vector{Float32}(undef, lambda_n)
    val_nonzero = Vector{Float32}(undef, lambda_n)
    epochs = Vector{Int}(undef, lambda_n)
    train_loss_polish = Vector{Float32}(undef, lambda_n)
    val_loss_polish = Vector{Float32}(undef, lambda_n)
    epochs_polish = Vector{Int}(undef, lambda_n)
    val_loss_relax = Matrix{Float32}(undef, lambda_n, gamma_n)
    val_loss_se_relax = Matrix{Float32}(undef, lambda_n, gamma_n)

    # Loop over lambda values
    for i in 1:lambda_n

        # Print status update
        if verbose && i == 1
            print("Training with regularisation parameter $i of $lambda_n...\n")
        elseif verbose
            print("\r\033[1ATraining with regularisation parameter $i of $lambda_n...\n")
        end

        # Save lambda or initialise lambda_max
        if isnothing(lambda)
            lambda_i = Inf
        else
            lambda_i = lambda[i]
        end

        # Create neural network model
        if initialise == "cold" || i == 1
            model_i = gennet(hidden_layers, sign_constraint, dropout, p + intercept, m, 
                activation_fun)
        else
            model_i = deepcopy(model[i - 1])
        end

        # Move model to training device
        model_i = device(model_i)

        # Train model
        model_i, theta[i], train_loss[i], val_loss[i], val_loss_se[i], val_nonzero[i], 
            epochs[i] = train(model_i, lambda_i, x, z, y, x_val, z_val, y_val, loss, intercept, 
                group_info, optimiser, epoch_max, early_stop, patience, verbose, verbose_freq)

        # If no lambda, compute lambda
        if isnothing(lambda)
            lambda = range(sum(abs.(model_i(z)[1 + intercept:end, :])) / n, 0, lambda_n)
        end

        if relax

            # Compute masking matrices
            mask = threshold(model_i(z)[1 + intercept:end, :], theta[i], group_info) .!= 0
            mask_val = threshold(model_i(z_val)[1 + intercept:end, :], theta[i], group_info) .!= 0

            if initialise == "cold" || i == 1
                model_polish_i = gennet(hidden_layers, sign_constraint, dropout, p + intercept, m, 
                    activation_fun)
            else
                model_polish_i = deepcopy(model_polish[i - 1])
            end

            # Move polished model to training device
            model_polish_i = device(model_polish_i)

            # Train polished model
            model_polish_i, train_loss_polish[i], val_loss_polish[i], epochs_polish[i] = 
                train_polish(model_polish_i, mask, mask_val, x, z, y, x_val, z_val, y_val, loss, 
                    intercept, group_info, optimiser, epoch_max, early_stop, patience, verbose, 
                    verbose_freq)
            
            # Save validation losses for relaxed fits specified by gamma
            beta = model_i(z_val)
            beta_int = threshold(beta[1 + intercept:end, :], theta[i], group_info)
            beta_polish = model_polish_i(z_val)
            beta_int_polish = beta_polish[1 + intercept:end, :] .* (beta_int .!= 0)
            for j in 1:gamma_n
                beta_relax = device(zeros(p + intercept, n_val))
                beta_relax[1 + intercept:end, :] = (1 - gamma[j]) * beta_int + gamma[j] * 
                    beta_int_polish
                if intercept
                    beta_relax[1, :] = (1 - gamma[j]) * beta[1, :] + gamma[j] * beta_polish[1, :]
                end
                y_hat = sum(x_val .* beta_relax[1 + intercept:end, :], dims = 1)
                if intercept
                    y_hat += transpose(beta_relax[1, :])
                end
                val_loss_relax[i, j] = loss(y_hat, y_val)
                val_loss_se_relax[i, j] = loss(y_hat, y_val, agg = x -> 
                    Statistics.std(x, corrected = false)) / sqrt(n_val)
            end

            # Move polished model back to CPU
            model_polish[i] = Flux.cpu(model_polish_i)

        end

        # Move model back to CPU
        model[i] = Flux.cpu(model_i)

    end

    # Save min and 1se values of lambda and gamma
    if relax
        index_min = argmin(val_loss_relax)
        index_1se = findall(val_loss_relax .<= val_loss_relax[index_min] + 
            val_loss_se_relax[index_min])
        index_1se = sort(index_1se, by = i -> (lambda[i[1]], gamma[i[2]]))[1]
        lambda_min = lambda[index_min[1]]
        lambda_1se = lambda[index_1se[1]]
        gamma_min = gamma[index_min[2]]
        gamma_1se = gamma[index_1se[2]]
    else
        index_min = argmin(val_loss)
        index_1se = findall(val_loss .<= val_loss[index_min] + val_loss_se[index_min])
        index_1se = sort(index_1se, by = i -> lambda[i])[1]
        lambda_min = lambda[index_min]
        lambda_1se = lambda[index_1se]
        gamma_min = 0.0
        gamma_1se = 0.0
    end

    # Set type to ContextualLassoFit
    ContextualLassoFit(model, model_polish, lambda, gamma, intercept, group_info, relax, lambda_min, 
        gamma_min, lambda_1se, gamma_1se, theta, x_mean, x_sd, z_mean, z_sd, y_mean, y_sd, 
        train_loss, val_loss, val_loss_se, val_nonzero, epochs, train_loss_polish, val_loss_polish, 
        epochs_polish, val_loss_relax, val_loss_se_relax)

end

#==================================================================================================#
# Function to extract coefficients from a fitted contextually sparse linear model
#==================================================================================================#

"""
    coef(fit::ContextualLassoFit, z; <keyword arguments>)

Produce coefficients from a contextual lasso fit using contextual features `z`. Set \
`lambda = "lambda_min"` for predictions from the model with minimum validation loss or \
`lambda = "lambda_1se"` for predictions from the most regularized model within one standard error \
of the minimum. Alternatively, specify a value of `lambda` for predictions from a particular \
model. The relaxation parameter `gamma` can be configured similarly.

See also [`fit`](@ref), [`predict`](@ref), [`plot`](@ref).
"""
function coef(fit::ContextualLassoFit, z::Matrix{<:Real}; 
    lambda::Union{String, Real} = "lambda_1se", gamma::Union{String, Real} = "gamma_1se")

    # Validate arguments
    if isa(lambda, String)
        lambda in ["lambda_min", "lambda_1se"] || 
        error("""lambda should be "lambda_min" or "lambda_1se".""")
    end
    if isa(gamma, String)
        gamma in ["gamma_min", "gamma_1se"] || 
        error("""gamma should be "gamma_min" or "gamma_1se".""")
    end

    # Flux defaults to f32 model parameters
    z = Flux.f32(z)

    # Standardise contextual features before they enter neural net
    z = (z .- fit.z_mean) ./ fit.z_sd

    # Transpose because Flux expects features in rows and samples in columns
    z = transpose(z)

    # Find correct lambda index
    if lambda == "lambda_min"
        index_lambda = findall(fit.lambda .== fit.lambda_min)[1]
    elseif lambda == "lambda_1se"
        index_lambda = findall(fit.lambda .== fit.lambda_1se)[1]
    else
        index_lambda = argmin(abs.(fit.lambda .- lambda))
    end

    # Compute coefficients by performing a forward pass
    beta = fit.model[index_lambda](z)

    # Project coefficients onto l1 ball
    beta[1 + fit.intercept:end, :] = threshold(beta[1 + fit.intercept:end, :], 
        fit.theta[index_lambda], fit.group_info)

    # Transform back to original orientation
    beta = permutedims(beta)

    # Express coefficients on original scale
    function unstandardise!(beta)
        if fit.intercept
            beta[:, 2:end] = beta[:, 2:end] ./ fit.x_sd .* fit.y_sd
            beta[:, 1] = beta[:, 1] .* fit.y_sd .- beta[:, 2:end] * transpose(fit.x_mean) .+ 
            fit.y_mean
        else
            beta .= beta ./ fit.x_sd .* fit.y_sd
        end
    end
    unstandardise!(beta)

    # If relax, combine original and polished coefficients
    if !fit.relax
        beta
    else
        if gamma == "gamma_min"
            index_gamma = findall(fit.gamma .== fit.gamma_min)[1]
        elseif gamma == "gamma_1se"
            index_gamma = findall(fit.gamma .== fit.gamma_1se)[1]
        else
            index_gamma = argmin(abs.(fit.gamma .- gamma))
        end
        beta_polish = permutedims(fit.model_polish[index_lambda](z)) .* (beta .!= 0)
        unstandardise!(beta_polish)
        (1 - fit.gamma[index_gamma]) * beta + fit.gamma[index_gamma] * beta_polish
    end

end

#==================================================================================================#
# Function to produce predictions from a fitted contextually sparse linear model
#==================================================================================================#

"""
    predict(fit::ContextualLassoFit, x, z; <keyword arguments>)

Produce predictions from a contextual lasso fit using explanatory features `x` and contextual \
features `z`. Set `lambda = "lambda_min"` for predictions from the model with minimum validation \
loss or `lambda = "lambda_1se"` for predictions from the most regularized model within one \
standard error of the minimum. Alternatively, specify a value of `lambda` for predictions from a \
particular model. The relaxation parameter `gamma` can be configured similarly.

See also [`fit`](@ref), [`coef`](@ref), [`plot`](@ref).
"""
function predict(fit::ContextualLassoFit, x::Matrix{<:Real}, z::Matrix{<:Real}; 
    lambda::Union{String, Real} = "lambda_1se", gamma::Union{String, Real} = "gamma_1se")

    # Compute prediction x * beta(z)
    beta = coef(fit, z, lambda = lambda, gamma = gamma)
    yhat = sum(x .* beta[:, 1 + fit.intercept:end], dims = 2)
    if fit.intercept
        yhat += beta[:, 1]
    end
    dropdims(yhat, dims = 2)

end

#==================================================================================================#
# Function to plot the validation curve from a fitted contextually sparse linear model
#==================================================================================================#

"""
    plot(fit::ContextualLassoFit)

Produce a plot of validation loss from a contextual lasso fit as a function of the regularisation \
parameter. Labels indicate number of nonzero coefficients over the validation data. Grey vertical \
lines indicate the minimum point on the curve and the point on the curve within one standard error \
of the minimum.

See also [`fit`](@ref), [`coef`](@ref), [`predict`](@ref).
"""
function plot(fit::ContextualLassoFit)

    if !fit.relax
    
        # If no relax, plot validation curve with points and error bars
        Gadfly.plot(
            x = fit.lambda,
            y = fit.val_loss,
            ymin = fit.val_loss - fit.val_loss_se,
            ymax = fit.val_loss + fit.val_loss_se,
            label = string.(round.(fit.val_nonzero, digits = 1)),
            xintercept = [fit.lambda_min, fit.lambda_1se],
            Gadfly.Geom.point,
            Gadfly.Geom.line,
            Gadfly.Geom.yerrorbar,
            Gadfly.Geom.label,
            Gadfly.Geom.vline(color = "grey"),
            Gadfly.Guide.xlabel("λ"),
            Gadfly.Guide.ylabel("Validation loss")
            )

    else

        # If relax, plot simplified validation curve without points or error bars and color by
        # relaxation parameter
        Gadfly.plot(
            x = repeat(fit.lambda, length(fit.gamma)),
            y = vec(fit.val_loss_relax),
            color = repeat(fit.gamma, inner = length(fit.lambda)),
            label = string.(round.(fit.val_nonzero, digits = 1)),
            xintercept = [fit.lambda_min, fit.lambda_1se],
            Gadfly.Geom.line,
            Gadfly.Geom.label,
            Gadfly.Geom.vline(color = "grey"),
            Gadfly.Guide.xlabel("λ"),
            Gadfly.Guide.ylabel("Validation loss"),
            Gadfly.Guide.colorkey("γ")
            )

    end

end

end