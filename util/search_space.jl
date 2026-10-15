#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#
# Hyperparameter search space of the PeN-ODE and NeuralFMU studies.
# The shuffled order defines the run ids: combs[n] are the hyperparameters of run_n.
#

using Random
using Lux # relu, sigmoid

num_neurons_list = [8, 16, 32]
num_hidden_list = [1, 2, 3]
type_act_list = [tanh, relu, sigmoid]
act_last_list = [tanh]
initial_gate_list = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]
loss_discr_list = [1e-4]
training_sample_length_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
learning_rate = [1e-2, 1e-3, 1e-4]
beta_1 = [0.99, 0.9]
beta_2 = [0.9999, 0.999, 0.99]

# entries: (num_neurons, num_hidden, type_act, act_last, initial_gate, loss_discretization,
#           training_sample_length, learning_rate, beta_1, beta_2)
combs = collect(Iterators.product(num_neurons_list, num_hidden_list, type_act_list, act_last_list, initial_gate_list, loss_discr_list, training_sample_length_list, learning_rate, beta_1, beta_2))
shuffle!(MersenneTwister(1234), combs)
