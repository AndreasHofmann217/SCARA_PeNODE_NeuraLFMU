#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#
# Helpers for the seed study: run directories, validation history and best checkpoint of a run.
#

using JLD2

const SEEDS = [1, 12, 123, 1234, 12345]
const SEED_CONFIGS = (PeNODE=[54, 55, 83], NeuralFMU=[74, 82, 86])
const VALID_INTERVAL = 50 # validation loss is calculated every 50 steps (see util/setup_train.jl)

seed_run_dir(results_dir, arch_dir, config, seed) = joinpath(results_dir, arch_dir, "run_$(config)_$(seed)")

# validation MAE history of a run: every VALID_INTERVAL steps, plus the final evaluation after the last step
# (losses.jld2 of a finished run, otherwise the latest checkpoint)
function load_validation_history(rundir)
    lossfile = joinpath(rundir, "losses.jld2")
    if isfile(lossfile)
        v = JLD2.load(lossfile, "valid_losses")
        laststep = VALID_INTERVAL * (length(v) - 1) # last entry = extra final evaluation
        complete = true
    else
        ckpts = filter(f -> occursin(r"^parameters\d+\.jld2$", f), readdir(rundir))
        isempty(ckpts) && error("no checkpoints in $(rundir)")
        laststep = maximum(parse(Int, match(r"\d+", f).match) for f in ckpts)
        v = JLD2.load(joinpath(rundir, "parameters$(laststep).jld2"), "valid_losses")
        complete = false
    end
    return (valid_losses=v, laststep=laststep, complete=complete)
end

# best checkpoint w.r.t. the validation MAE (the final evaluation belongs to the last checkpoint)
function best_checkpoint(rundir)
    h = load_validation_history(rundir)
    i = argmin(h.valid_losses)
    return (step=min(VALID_INTERVAL * i, h.laststep), valid_loss=h.valid_losses[i],
        final_valid_loss=h.valid_losses[end], laststep=h.laststep, complete=h.complete)
end
