#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using MiniPeNODETrainingFramework
using JLD2
using Statistics
using ReverseDiff
using Optimisers
using SciMLSensitivity

include("../ANN_Architectures/OOM_Architecture.jl")
include("../models/PeNODE_SCARA.jl")


# TCP position from the generated observation function of the model (uses the model parameters of tm,
# e.g. modified arm lengths). Only the TCP entries are kept per time step, since the full observable vector
# (calc_..._obs) of every time step would need a lot of memory.
# friction = true additionally returns the TCP friction force (PeN-ODE: output of the ANN; not available for the
# NeuralFMU, whose observation function has no ANN friction force)
function calcTCPObservables(sol, (obs_fun, obs_names, obs_name_dict), tm, meas; friction::Bool=false)
    # the observation function reads the inputs and the ANN via se.penode_comp
    penode_comp = deepcopy(tm.penode_comp)
    penode_comp.input_fun = MiniPeNODETrainingFramework.getInputFunctions(meas, penode_comp.input_signals)
    se = (penode_comp=penode_comp,)

    quantities = ["tcp_x" => "scara₊tcp₊x(t)", "tcp_y" => "scara₊tcp₊y(t)"]
    friction && append!(quantities, ["friction_x" => "scara₊tcp₊fx(t)", "friction_y" => "scara₊tcp₊fy(t)"])
    idx = [obs_name_dict[name] for (_, name) in quantities]
    values = zeros(length(sol.t), length(quantities))
    for i in eachindex(sol.t)
        obs = obs_fun(sol.u[i], penode_comp.penode_parameters, sol.t[i], tm.p_flat, se)
        values[i, :] = obs[idx]
    end
    result = Dict{String,Any}("time" => sol.t)
    for (j, (key, _)) in enumerate(quantities)
        result[key] = values[:, j]
    end
    return result
end


function simulateTrainedPeNODE(input_signals, meas_model_mapping, loss_selection, hyper_params, runid, training_data, validation_data, test_data; parameterid=5000, customdir::Union{Nothing,String}=nothing, custommodelparams::Bool=false, return_test_solution::Bool=custommodelparams)

    # parameters
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


    # initialize ANN architecture before updating parameters
    rng = Xoshiro(1234)
    p_flat, nn_fun = deriveMLModelOOM(num_neurons, num_hidden, type_act, act_last, initial_gate, training_data; rng=rng)



    # define TrainingManager components
    penode_comp = PENODE_COMPONENTS(
        get_SCARAPeNODE_rhs(),
        custommodelparams ? get_SCARAPeNODE_parametersLengthMod()[1] : get_SCARAPeNODE_parameters()[1],
        get_SCARAPeNODE_mass_matrix(),
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


    if isnothing(customdir)
        params = JLD2.load("./TrainingResults/PeNODE/run_$(runid)/parameters$(parameterid).jld2")["parameters"]
    else
        run_dir = joinpath(customdir, "parameters$(parameterid).jld2")
        params = JLD2.load(run_dir)["parameters"]
    end

    tm.p_flat = params

    # calculation of losses
    valid_loss = calculateLoss(tm, tm.validation_meas[1]; t_stops=tm.validation_meas[1].signals["inp3"].time)
    println("validation_loss: $(valid_loss)")

    tm_test = deepcopy(tm)
    tm_test.validation_meas = test_data
    test_loss = calculateLoss(tm_test, tm_test.validation_meas[1]; t_stops=tm_test.validation_meas[1].signals["inp3"].time)
    println("test_loss: $(test_loss)")

    if return_test_solution
        # simulation of the test trajectory (test_data), e.g. for comparing against the reference
        test_solution = simulateMeasurements(tm_test, tm_test.validation_meas; t_stops=tm_test.validation_meas[1].signals["inp3"].time)[1][1]
        test_observables = calcTCPObservables(test_solution, get_SCARAPeNODE_obs(), tm_test, test_data[1]; friction=true)
    else
        test_solution = nothing
        test_observables = nothing
    end

    return (valid_loss, test_loss, test_solution, test_observables)

end


