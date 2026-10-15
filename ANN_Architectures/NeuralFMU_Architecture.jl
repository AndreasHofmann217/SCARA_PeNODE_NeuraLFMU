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

function deriveMLModelNeuralFMU(num_neurons::Int64, num_hidden::Int64, type_act::Function, act_last::Function, initial_gate::Union{Float64,Vector{Float64}}, training_data::Vector{Measurement}; rng=Random.default_rng())

    # input_signals = [
    #     "scara₊rev2₊phi(t)",
    #     "scara₊rev1₊phi(t)",
    #     "motor2₊inductor₊i(t)",
    #     "scara₊rev1₊w(t)",
    #     "scara₊rev2₊w(t)",
    #     "motor1₊inductor₊i(t)",
    #     "der_scara₊rev2₊phi(t)",
    #     "der_scara₊rev1₊phi(t)",
    #     "der_motor2₊inductor₊i(t)",
    #     "der_scara₊rev1₊w(t)",
    #     "der_scara₊rev2₊w(t)",
    #     "der_motor1₊inductor₊i(t)",
    #     "TCP.x",
    #     "TCP.y",
    #     "TCP.vx",
    #     "TCP.vy",
    #     "inp3"
    # ]

    ### diss tobi
    input_signals = [
        "scara₊rev2₊phi(t)",
        "scara₊rev1₊phi(t)",
        "der_scara₊rev1₊w(t)",
        "der_scara₊rev2₊w(t)",
        "TCP.vx",
        "TCP.vy",
        "inp3"
    ]


    network_chain = []
    push!(network_chain, Dense(length(input_signals), num_neurons, type_act))
    for i in 1:num_hidden
        push!(network_chain, Dense(num_neurons, num_neurons, type_act))
    end
    push!(network_chain, Dense(num_neurons, 2, act_last)) # we explicitly chose tanh for the last activation since it will restrict output to -1...+1 and is good base for the normalization whill will be added next

    inputs_mean = [mean(reduce(vcat, meas.signals[sig].value for meas in training_data)) for sig in input_signals]
    inputs_std = [Statistics.std(reduce(vcat, meas.signals[sig].value for meas in training_data)) for sig in input_signals]
    pushfirst!(network_chain, MeanStdNormalizerLayer(inputs_mean, inputs_std))

    # we output state derivatives!
    output_signals = [
        "der_scara₊rev1₊w(t)",
        "der_scara₊rev2₊w(t)"
    ]
    outputs_mean = [mean(reduce(vcat, meas.signals[sig].value for meas in training_data)) for sig in output_signals]
    outputs_std = [Statistics.std(reduce(vcat, meas.signals[sig].value for meas in training_data)) for sig in output_signals]
    push!(network_chain, MeanStdUnNormalizerLayer(outputs_mean, outputs_std))

    ANN = Lux.Chain(network_chain...)

    lux_params, lux_states = Lux.setup(rng, ANN) |> f64 # we want float64 precision, since we are doing simulation

    # we decouple parameters and network, so we can add additional quantities such as gates to the set of parameters
    p_ANN, restruct = Optimisers.destructure(lux_params)

    # add the parameters of the gates to the total set of parameters
    total_params = ComponentArray{Float64}(gate=deepcopy(initial_gate), p_ANN=p_ANN)

    # our function that will be called from se.penode_comp.nn_fun[1]
    # function penode_NN(phi2, phi1, i2, w1, w2, i1, d_phi2, d_phi1, d_i2, d_w1, d_w2, d_i1,
    #     tcp_x, tcp_y,  tcp_vx, tcp_vy, F_z, trainable_p)

    #     ann_out = Lux.apply(ANN,[phi2, phi1, i2, w1, w2, i1, d_phi2, d_phi1, d_i2, d_w1, d_w2, d_i1,
    #         tcp_x, tcp_y,  tcp_vx, tcp_vy, F_z],restruct(trainable_p.p_ANN),lux_states)[1]
    #     return trainable_p.gate .* ann_out + (1.0 .- trainable_p.gate) .* [d_w1, d_w2]
    # end

    function neuralfmu_NN(phi2, phi1, d_w1, d_w2, tcp_vx, tcp_vy, F_z, trainable_p)

        ann_out = Lux.apply(ANN, [phi2, phi1, d_w1, d_w2, tcp_vx, tcp_vy, F_z], restruct(trainable_p.p_ANN), lux_states)[1]
        return trainable_p.gate .* ann_out + (1.0 .- trainable_p.gate) .* [d_w1, d_w2]
    end


    # array of functions that will bes tored at se.penode_comp.nn_fun
    nn_fun = Vector{Function}([neuralfmu_NN])

    return total_params, nn_fun

end
