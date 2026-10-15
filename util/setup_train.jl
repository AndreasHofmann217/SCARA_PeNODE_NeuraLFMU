#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using MiniPeNODETrainingFramework
using ScaraModel
using ModelingToolkit
using Statistics
using Random
using Optimisers
using ComponentArrays
using ReverseDiff
using SciMLSensitivity
using Lux
using JLD2

# generate nn_fun and parameters
include("./../ANN_Architectures/OOM_Architecture.jl") ## OOM model
include("./../ANN_Architectures/NeuralFMU_Architecture.jl") ## Neural FMU model

# load actual model
include("./../models/PeNODE_SCARA.jl")
include("./../models/NeuralFMU_SCARA.jl")


function setupAndTrain(architecture, runid, hyper_params, training_data, validation_data;
    custom_rng::Union{Nothing,Integer}=nothing, number_of_steps=5000, continueExisting::Union{Nothing,String}=nothing)

    println("Running Model $(runid) with architecture $(architecture)")
    ## unpack hyperparameters
    num_neurons = hyper_params[1]
    num_hidden = hyper_params[2]
    type_act = hyper_params[3]
    act_last = hyper_params[4]
    initial_gate = hyper_params[5]
    loss_discretization = hyper_params[6]
    training_sample_length = hyper_params[7]
    learning_rate = hyper_params[8]
    beta_1 = hyper_params[9]
    beta_2 = hyper_params[10]


    rng = isnothing(custom_rng) ? Xoshiro(1234) : Xoshiro(custom_rng)
    if architecture == :OOM
        p_flat, nn_fun = deriveMLModelOOM(num_neurons, num_hidden, type_act, act_last, initial_gate, training_data; rng=rng)
        get_rhs, get_parameters, get_mass_matrix = get_SCARAPeNODE_rhs, get_SCARAPeNODE_parameters, get_SCARAPeNODE_mass_matrix
    elseif architecture == :NeuralFMU
        p_flat, nn_fun = deriveMLModelNeuralFMU(num_neurons, num_hidden, type_act, act_last, initial_gate, training_data; rng=rng)
        get_rhs, get_parameters, get_mass_matrix = get_ScaraNeuralFMU_rhs, get_ScaraNeuralFMU_parameters, get_ScaraNeuralFMU_mass_matrix
    else
        println("invalid architecture selection! Use either :OOM or :NeuralFMU")
        throw(ArgumentError("invalid architecture selection"))
    end

    # define TrainingManager components
    penode_comp = PENODE_COMPONENTS(
        get_rhs(),
        get_parameters()[1],
        get_mass_matrix(),
        input_signals,
        nn_fun
    )

    t_stops = training_data[1].signals["inp3"].time
    sol_comp = SOLVER_COMPONENTS(
        Rodas5P(autodiff=false),
        1e-6,
        1e-6,
        2e7,
        t_stops,
        nothing
    )

    grad_comp = GRAD_COMPONENTS(
        ReverseDiffAdjoint(),
        ReverseDiff.gradient!
    )

    opt_rule = Optimisers.Adam(learning_rate, (beta_1, beta_2))
    opt_comp = OPT_COMPONENTS(
        opt_rule,
        Optimisers.setup(opt_rule, p_flat)
    )

    meas_model_mapping = [("scara₊rev1₊phi(t)", 2), ("scara₊rev2₊phi(t)", 1), ("scara₊rev1₊w(t)", 4), ("scara₊rev2₊w(t)", 5), ("motor1₊inductor₊i(t)", 9), ("motor2₊inductor₊i(t)", 3)]
    loss_selection = [3, 9]

    loss_comp = LOSS_COMPONENTS(
        loss_MAE,
        meas_model_mapping,
        loss_selection,
        loss_discretization
    )

    tm = TrainingManager(
        training_data,
        validation_data,
        p_flat,
        penode_comp,
        loss_comp,
        grad_comp,
        sol_comp,
        opt_comp,
        nothing
    )





    tm.training_samples = generateSamples(tm.train_meas, training_sample_length, tm.loss_comp.sampling_rate, tm.loss_comp.loss_signal_mapping, tm.loss_comp.loss_idx; allow_short_batches=true)

    valid_loss = calculateLoss(tm, tm.validation_meas[1]; t_stops=tm.validation_meas[1].signals["inp3"].time)
    println("Initial validation loss: $(valid_loss)")

    losses = Float64[]
    valid_losses = Float64[]

    run_dir = isnothing(custom_rng) ? "run_$(runid)" : "run_$(runid)_$(custom_rng)"

    start_step = 1

    # load parameters and optimizer settings
    paramnum(s) = (m = match(r"parameters(\d+)\.jld2$", basename(s))) === nothing ? nothing : parse(Int, m[1])
    if !isnothing(continueExisting)
        jld2_set = JLD2.load(continueExisting)
        losses = jld2_set["losses"]
        valid_losses = jld2_set["valid_losses"]
        tm.p_flat = jld2_set["parameters"]
        tm.opt_comp.optimizer_state = jld2_set["opt_state"]
        start_step = paramnum(continueExisting) + 1 # checkpoint is saved after this step was completed
        println("Continuing from $(continueExisting) at step $(start_step)")
    end



    for i in start_step:number_of_steps
        println("step $(i)")
        _loss, _grad = doStep(tm)
        println("sample loss: $(_loss)")
        push!(losses, _loss)
        if i%50 == 0
            valid_loss = calculateLoss(tm, tm.validation_meas[1]; t_stops=tm.validation_meas[1].signals["inp3"].time)
            println("Current validation loss: $(valid_loss)")
            push!(valid_losses, valid_loss)

            JLD2.save("./$(architecture)/$(run_dir)/parameters$(i).jld2", Dict("parameters" => tm.p_flat, "opt_state" => tm.opt_comp.optimizer_state, "losses" => losses, "valid_losses" => valid_losses))
        end

    end

    # final validation_loss
    final_valid_loss = calculateLoss(tm, tm.validation_meas[1]; t_stops=tm.validation_meas[1].signals["inp3"].time)
    push!(valid_losses, final_valid_loss)
    JLD2.save("./$(architecture)/$(run_dir)/losses.jld2", Dict("losses" => losses, "valid_losses" => valid_losses))
    println("Final validation loss of run $(runid) with architecture $(architecture): $(final_valid_loss)")

end
