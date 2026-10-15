#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Lux
using MiniPeNODETrainingFramework
using Random
using Statistics
using Optimisers
using ComponentArrays


function deriveMLModelOOM(num_neurons::Int64, num_hidden::Int64, type_act::Function, act_last::Function, initial_gate::Union{Float64,Vector{Float64}}, training_data::Vector{Measurement}; rng=Random.default_rng())

    network_chain = []
    # 1. Add normal MLP layers
    push!(network_chain, Dense(3, num_neurons, type_act))
    for i in 1:num_hidden
        push!(network_chain, Dense(num_neurons, num_neurons, type_act))
    end
    push!(network_chain, Dense(num_neurons, 2, act_last)) # we explicitly chose tanh for the last activation since it will restrict output to -1...+1 and is good base for the normalization whill will be added next

    # 2. determine mean and std for normalization layers and add to chain
    input_signals = ["TCP.vx", "TCP.vy", "inp3"]

    inputs_mean = [mean(reduce(vcat, meas.signals[sig].value for meas in training_data)) for sig in input_signals]
    inputs_std = [Statistics.std(reduce(vcat, meas.signals[sig].value for meas in training_data)) for sig in input_signals]
    pushfirst!(network_chain, MeanStdNormalizerLayer(inputs_mean, inputs_std))

    output_signals = ["TCP.FrictionForceX", "TCP.FrictionForceY"]

    outputs_mean = [mean(reduce(vcat, meas.signals[sig].value for meas in training_data)) for sig in output_signals]
    outputs_std = [Statistics.std(reduce(vcat, meas.signals[sig].value for meas in training_data)) for sig in output_signals]
    push!(network_chain, MeanStdUnNormalizerLayer(outputs_mean, outputs_std))

    # 3. build actual Lux network
    ANN = Lux.Chain(network_chain...)

    lux_params, lux_states = Lux.setup(rng, ANN) |> f64 # we want float64 precision, since we are doing simulation

    # we decouple parameters and network, so we can add additional quantities such as gates to the set of parameters
    p_ANN, restruct = Optimisers.destructure(lux_params)

    # add the parameters of the gates to the total set of parameters
    total_params = ComponentArray{Float64}(gate=deepcopy(initial_gate), p_ANN=p_ANN)

    # our function that will be called from se.penode_comp.nn_fun[1]
    function penode_NN(vx, vy, Fnormal, trainable_p)
        return trainable_p.gate .* Lux.apply(ANN, [vx, vy, Fnormal], restruct(trainable_p.p_ANN), lux_states)[1]
    end

    # array of functions that will bes tored at se.penode_comp.nn_fun
    nn_fun = Vector{Function}([penode_NN])

    return total_params, nn_fun


end