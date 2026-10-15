#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#
# Structural adaptation: trained PeN-ODE and NeuralFMU (unchanged ANN parameters) with the arm lengths of the
# physical model changed from L1 = 0.2 m, L2 = 0.1 m to L1 = 0.25 m, L2 = 0.15 m
# (get_..._parametersLengthMod in models/), tested against the reference of the SCARA with the same lengths.
#   1. test MAE with original and modified arm lengths for the best validation run of each architecture
#   2. export of motor currents, TCP position and TCP friction force (reference and models) to JLD2
#
# usage (in the repository root): julia --project=. analysis_structural_adaptation.jl
# requires the reference data in data/ incl. ScaraTest_L1_025_L2_015.h5 (see generate_data.jl)
#

using MiniPeNODETrainingFramework
using JLD2

include("util/search_space.jl")
include("util/seed_runs.jl")
include("util/penode_postprocess.jl")
include("util/neuralfmu_postprocess.jl")

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
# results of the seed study (see analysis_seeds.jl)
const RESULTS_DIR = joinpath(@__DIR__, "seed_study_results")
const ARCH_DIRS = (PeNODE="PeNODE", NeuralFMU="NeuralFMU")

const LENGTHS_ORIGINAL = (L1=0.2, L2=0.1)
const LENGTHS_MODIFIED = (L1=0.25, L2=0.15) # must match get_..._parametersLengthMod and the reference data

const EXPORT_FILE = joinpath(@__DIR__, "structural_adaptation_trajectories.jld2")

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
input_signals = ["inp1", "inp2", "inp3"]
# make sure mapping order corresponds to states in generated model
state_mapping = ["scara₊rev2₊phi(t)", "scara₊rev1₊phi(t)", "motor2₊inductor₊i(t)", "scara₊rev1₊w(t)", "scara₊rev2₊w(t)", nothing, nothing, nothing, "motor1₊inductor₊i(t)"]
training_data = generateMeasurementFromHDF5(joinpath(@__DIR__, "data", "ScaraTrain.h5"), state_mapping; measurement_selection=["train"])
validation_data = generateMeasurementFromHDF5(joinpath(@__DIR__, "data", "ScaraValidate.h5"), state_mapping; measurement_selection=["validate"])
test_data = generateMeasurementFromHDF5(joinpath(@__DIR__, "data", "ScaraTest.h5"), state_mapping; measurement_selection=["test"])
lengthmod_test_data = generateMeasurementFromHDF5(joinpath(@__DIR__, "data", "ScaraTest_L1_025_L2_015.h5"), state_mapping; measurement_selection=["test"])

# loss on the motor currents, as in training (util/setup_train.jl)
meas_model_mapping = [("scara₊rev1₊phi(t)", 2), ("scara₊rev2₊phi(t)", 1), ("scara₊rev1₊w(t)", 4), ("scara₊rev2₊w(t)", 5), ("motor1₊inductor₊i(t)", 9), ("motor2₊inductor₊i(t)", 3)]
loss_selection = [3, 9]

# ---------------------------------------------------------------------------
# 1. best validation run of each architecture: original vs. modified arm lengths
# ---------------------------------------------------------------------------
function best_seed_run(arch)
    candidates = [(config=c, seed=s, rundir=seed_run_dir(RESULTS_DIR, ARCH_DIRS[arch], c, s))
                  for c in SEED_CONFIGS[arch] for s in SEEDS]
    bests = [best_checkpoint(r.rundir) for r in candidates]
    i = argmin([b.valid_loss for b in bests])
    return (candidates[i]..., step=bests[i].step, valid_loss=bests[i].valid_loss)
end

results = Dict{Symbol,Any}()
for (arch, simulate) in ((:PeNODE, simulateTrainedPeNODE), (:NeuralFMU, simulateTrainedNeuralFMU))
    selected = best_seed_run(arch)
    println("\n$(arch): best validation run config $(selected.config), seed $(selected.seed), step $(selected.step) (val. MAE $(selected.valid_loss))")
    args = (input_signals, meas_model_mapping, loss_selection, combs[selected.config], selected.config, training_data, validation_data)

    # baseline: original arm lengths, original test data
    (_, test_orig, sol_orig, obs_orig) = simulate(args..., test_data;
        parameterid=selected.step, customdir=selected.rundir, return_test_solution=true)
    # structural adaption: modified arm lengths in model and test data
    # (the returned validation loss compares the modified model against the original validation data and is not meaningful here)
    (_, test_mod, sol_mod, obs_mod) = simulate(args..., lengthmod_test_data;
        parameterid=selected.step, customdir=selected.rundir, custommodelparams=true)

    results[arch] = (run=selected, test_orig=test_orig, test_mod=test_mod,
        sol_orig=sol_orig, obs_orig=obs_orig, sol_mod=sol_mod, obs_mod=obs_mod)

    println("  test MAE original lengths (L1 = $(LENGTHS_ORIGINAL.L1) m, L2 = $(LENGTHS_ORIGINAL.L2) m):  $(round(test_orig, digits=6))")
    println("  test MAE modified lengths (L1 = $(LENGTHS_MODIFIED.L1) m, L2 = $(LENGTHS_MODIFIED.L2) m): $(round(test_mod, digits=6))")
    println("  change: $(round(100 * (test_mod / test_orig - 1), digits=1)) %")
end

# ---------------------------------------------------------------------------
# 2. export: motor currents, TCP position and (reference, PeN-ODE) TCP friction force on the test trajectory
# state order of the generated models: 1 rev2.phi, 2 rev1.phi, 3 motor2 current, ..., 9 motor1 current
# TCP position and friction force of the models from the generated observation functions (calcTCPObservables);
# the NeuralFMU has no explicit friction force
# ---------------------------------------------------------------------------
function hybrid_trajectory(sol, obs)
    result = Dict{String,Any}(
        "time" => sol.t,
        "motor1_current" => [u[9] for u in sol.u],
        "motor2_current" => [u[3] for u in sol.u],
    )
    for key in ("tcp_x", "tcp_y", "friction_x", "friction_y")
        haskey(obs, key) && (result[key] = obs[key])
    end
    return result
end

function reference_trajectory(meas)
    s = meas.signals
    t = s["TCP.x"].time
    return Dict(
        "time" => t,
        "motor1_current" => s["motor1₊inductor₊i(t)"].value,
        "motor2_current" => s["motor2₊inductor₊i(t)"].value,
        "tcp_x" => s["TCP.x"].value,
        "tcp_y" => s["TCP.y"].value,
        "friction_x" => s["TCP.FrictionForceX"].value,
        "friction_y" => s["TCP.FrictionForceY"].value,
        "target_x" => s["inp1"].interpolation.(t), # desired TCP trajectory (model inputs)
        "target_y" => s["inp2"].interpolation.(t),
    )
end

p, f = results[:PeNODE], results[:NeuralFMU]
JLD2.save(EXPORT_FILE, Dict(
    "reference_original" => reference_trajectory(test_data[1]),
    "reference_lengthmod" => reference_trajectory(lengthmod_test_data[1]),
    "penode_original" => hybrid_trajectory(p.sol_orig, p.obs_orig),
    "penode_lengthmod" => hybrid_trajectory(p.sol_mod, p.obs_mod),
    "neuralfmu_original" => hybrid_trajectory(f.sol_orig, f.obs_orig),
    "neuralfmu_lengthmod" => hybrid_trajectory(f.sol_mod, f.obs_mod),
    "info" => Dict(
        "penode_run" => "PeN-ODE config $(p.run.config), seed $(p.run.seed), parameters$(p.run.step)",
        "neuralfmu_run" => "NeuralFMU config $(f.run.config), seed $(f.run.seed), parameters$(f.run.step)",
        "lengths_original" => LENGTHS_ORIGINAL,
        "lengths_lengthmod" => LENGTHS_MODIFIED,
        "test_mae" => Dict("penode_original" => p.test_orig, "penode_lengthmod" => p.test_mod,
            "neuralfmu_original" => f.test_orig, "neuralfmu_lengthmod" => f.test_mod),
    ),
))
println("\nsaved: ", EXPORT_FILE)
