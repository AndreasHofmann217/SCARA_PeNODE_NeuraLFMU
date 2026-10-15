#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#
# Training of the PeN-ODE (:OOM) and NeuralFMU models
#   - hyperparameter study: the first 100 configurations of the shuffled search space, 5000 steps each
#   - seed study: the selected configurations with 5 different seeds, 15000 steps each
#
# usage (in the repository root): julia --project=. train.jl
# requires the reference data in data/ (see generate_data.jl)
#
# results (relative to this directory, see util/setup_train.jl):
#   <architecture>/run_<runid>/parameters<step>.jld2          (hyperparameter study)
#   <architecture>/run_<runid>_<seed>/parameters<step>.jld2   (seed study)
# every 50 steps, plus losses.jld2 at the end
#

using Distributed
using Random

cd(@__DIR__) # checkpoints are written relative to the working directory

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
const STUDY = :seeds # :hyperparameters or :seeds

const TOTAL_CORES = 10
const CORES_PER_WORKER = 2 # used for julia threads, parallel GC marking and BLAS threads

input_signals = ["inp1", "inp2", "inp3"]
# make sure mapping order corresponds to states in generated model
state_mapping = ["scara₊rev2₊phi(t)", "scara₊rev1₊phi(t)", "motor2₊inductor₊i(t)", "scara₊rev1₊w(t)", "scara₊rev2₊w(t)", nothing, nothing, nothing, "motor1₊inductor₊i(t)"]

# hyperparameter search space; the shuffled order defines the run ids (combs[n] -> run_n)
include("util/search_space.jl")

# ---------------------------------------------------------------------------
# runs: (architecture, runid, seed or nothing, checkpoint to continue from or nothing)
# seed = nothing -> network initialization with Xoshiro(1234), see setupAndTrain
# ---------------------------------------------------------------------------
const SEEDS = [1, 12, 123, 1234, 12345]

if STUDY == :hyperparameters
    runs = [(a, n, nothing, nothing) for a in (:OOM, :NeuralFMU) for n in 1:100]
    number_of_steps = 5000
elseif STUDY == :seeds
    runs = [(a, n, s, nothing) for (a, configs) in ((:OOM, [54, 55, 83]), (:NeuralFMU, [74, 82, 86])) for n in configs for s in SEEDS]
    number_of_steps = 15000
else
    throw(ArgumentError("invalid study selection, use :hyperparameters or :seeds"))
end

# output directories (JLD2.save does not create them)
for (a, n, s, _) in runs
    mkpath(joinpath(string(a), isnothing(s) ? "run_$(n)" : "run_$(n)_$(s)"))
end

# ---------------------------------------------------------------------------
# one worker process per run (own GC, no shared state); output is prefixed with "From worker N:"
# ---------------------------------------------------------------------------
addprocs(min(length(runs), TOTAL_CORES ÷ CORES_PER_WORKER);
    exeflags=["--project=$(Base.active_project())", "--threads=$(CORES_PER_WORKER)", "--gcthreads=$(CORES_PER_WORKER)"])

# load packages, model and data on the workers
@everywhere workers() begin
    using LinearAlgebra
    BLAS.set_num_threads($CORES_PER_WORKER) # default would be one BLAS thread per core in every process -> oversubscription
    include($(joinpath(@__DIR__, "util", "setup_train.jl")))
end
@everywhere workers() begin
    input_signals = $input_signals # global used inside setupAndTrain
    state_mapping = $state_mapping
    training_data = generateMeasurementFromHDF5($(joinpath(@__DIR__, "data", "ScaraTrain.h5")), state_mapping; measurement_selection=["train"])
    validation_data = generateMeasurementFromHDF5($(joinpath(@__DIR__, "data", "ScaraValidate.h5")), state_mapping; measurement_selection=["validate"])
end

# defined on all processes so pmap can reference it by name; each worker uses its own data globals
@everywhere function train_run((a, n, s, checkpoint, params, steps))
    try
        setupAndTrain(a, n, params, training_data, validation_data;
            number_of_steps=steps, custom_rng=s, continueExisting=checkpoint)
        return (architecture=a, runid=n, seed=s, checkpoint=checkpoint, status=:ok)
    catch e
        @warn "Run $n (seed $s) failed" exception = (e, catch_backtrace())
        return (architecture=a, runid=n, seed=s, checkpoint=checkpoint, status=e)
    end
end

# on_error keeps the other results if a worker process dies (e.g. out of memory)
run_results = pmap(train_run, [(a, n, s, c, combs[n], number_of_steps) for (a, n, s, c) in runs]; on_error=e -> e)
foreach(println, run_results)
rmprocs(workers())
