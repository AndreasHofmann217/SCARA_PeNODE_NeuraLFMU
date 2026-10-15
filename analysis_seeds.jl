#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#
# Analysis of the seed study (selected configurations, 5 seeds each, 15000 steps)
#   1. best and final validation MAE of every seed run (best checkpoint w.r.t. the validation MAE)
#   2. test MAE at the best checkpoint of every seed run (cached in seed_losses.csv)
#   3. figure: validation MAE over the training of all seed runs
#
# usage (in the repository root): julia --project=. analysis_seeds.jl
# requires the reference data in data/ (see generate_data.jl) when the test MAE is (re)computed
#

using MiniPeNODETrainingFramework
using Statistics
using CSV
using JLD2
using PGFPlotsX
using PGFPlotsX: Plot, Axis, Coordinates, Legend, LegendEntry, GroupPlot # explicit, PlotlyJS (loaded via MiniPeNODETrainingFramework) also exports Plot

include("util/search_space.jl")
include("util/seed_runs.jl")
include("util/penode_postprocess.jl")
include("util/neuralfmu_postprocess.jl")

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
# results of the seed study: run_<config>_<seed>/ with losses.jld2 and the checkpoints parameters<step>.jld2;
# for a new training with train.jl (STUDY = :seeds) use RESULTS_DIR = @__DIR__ and the PeN-ODE directory "OOM"
const RESULTS_DIR = joinpath(@__DIR__, "seed_study_results")
const ARCH_DIRS = (PeNODE="PeNODE", NeuralFMU="NeuralFMU")

const LOSS_FILE = joinpath(@__DIR__, "seed_losses.csv")
const RECOMPUTE = false # true: evaluate the test MAE of all best checkpoints again

const FIG_DIR = joinpath(@__DIR__, "figures")
mkpath(FIG_DIR)

# ---------------------------------------------------------------------------
# 1. + 2. best checkpoint, validation and test MAE of every seed run
# ---------------------------------------------------------------------------
function evaluate_seed_runs()
    input_signals = ["inp1", "inp2", "inp3"]
    # make sure mapping order corresponds to states in generated model
    state_mapping = ["scara₊rev2₊phi(t)", "scara₊rev1₊phi(t)", "motor2₊inductor₊i(t)", "scara₊rev1₊w(t)", "scara₊rev2₊w(t)", nothing, nothing, nothing, "motor1₊inductor₊i(t)"]
    training_data = generateMeasurementFromHDF5(joinpath(@__DIR__, "data", "ScaraTrain.h5"), state_mapping; measurement_selection=["train"])
    validation_data = generateMeasurementFromHDF5(joinpath(@__DIR__, "data", "ScaraValidate.h5"), state_mapping; measurement_selection=["validate"])
    test_data = generateMeasurementFromHDF5(joinpath(@__DIR__, "data", "ScaraTest.h5"), state_mapping; measurement_selection=["test"])

    # loss on the motor currents, as in training (util/setup_train.jl)
    meas_model_mapping = [("scara₊rev1₊phi(t)", 2), ("scara₊rev2₊phi(t)", 1), ("scara₊rev1₊w(t)", 4), ("scara₊rev2₊w(t)", 5), ("motor1₊inductor₊i(t)", 9), ("motor2₊inductor₊i(t)", 3)]
    loss_selection = [3, 9]

    rows = []
    for (arch, simulate) in ((:PeNODE, simulateTrainedPeNODE), (:NeuralFMU, simulateTrainedNeuralFMU))
        for config in SEED_CONFIGS[arch], seed in SEEDS
            rundir = seed_run_dir(RESULTS_DIR, ARCH_DIRS[arch], config, seed)
            best = best_checkpoint(rundir)
            println("evaluating $(arch) config $(config), seed $(seed), best checkpoint $(best.step)")
            (valid_loss, test_loss) = simulate(input_signals, meas_model_mapping, loss_selection, combs[config], config,
                training_data, validation_data, test_data; parameterid=best.step, customdir=rundir)
            push!(rows, (architecture=string(arch), config=config, seed=seed, steps=best.laststep, complete=best.complete,
                best_step=best.step, best_valid_loss=best.valid_loss, final_valid_loss=best.final_valid_loss,
                valid_loss=valid_loss, test_loss=test_loss))
        end
    end
    CSV.write(LOSS_FILE, identity.(rows))
end

(RECOMPUTE || !isfile(LOSS_FILE)) && evaluate_seed_runs()
results = collect(CSV.File(LOSS_FILE))

# valid_loss is the validation MAE recomputed at the best checkpoint and should equal best_valid_loss from training
println("\nmax |recomputed - logged| validation MAE at the best checkpoints: ",
    maximum(abs(r.valid_loss - r.best_valid_loss) for r in results))

stats(v) = "$(round(mean(v), digits=4)) ± $(round(std(v), digits=4)) (min $(round(minimum(v), digits=4)), max $(round(maximum(v), digits=4)))"

for arch in ("PeNODE", "NeuralFMU")
    println("\n=== $(arch) ===")
    println("config  seed   best step  best val. MAE  final val. MAE  test MAE (best)")
    for r in filter(r -> r.architecture == arch, results)
        println(rpad(r.config, 8), rpad(r.seed, 7), rpad(r.best_step, 11), rpad(round(r.best_valid_loss, digits=6), 15),
            rpad(round(r.final_valid_loss, digits=6), 16), round(r.test_loss, digits=6), r.complete ? "" : "  (incomplete)")
    end
    for config in SEED_CONFIGS[Symbol(arch)]
        rs = filter(r -> r.architecture == arch && r.config == config, results)
        println("config $(config):")
        println("  best val. MAE  ", stats([r.best_valid_loss for r in rs]))
        println("  final val. MAE ", stats([r.final_valid_loss for r in rs]))
        println("  test MAE       ", stats([r.test_loss for r in rs]))
    end
    best = argmin(r -> r.best_valid_loss, filter(r -> r.architecture == arch, results))
    println("best validation run: config $(best.config), seed $(best.seed), step $(best.best_step) ",
        "(val. MAE $(round(best.best_valid_loss, digits=6)), test MAE $(round(best.test_loss, digits=6)))")
end

# ---------------------------------------------------------------------------
# 3. figure: validation MAE over the training, one column per architecture, one row per configuration,
#    one curve per seed (thin: every validation, thick: moving average)
# ---------------------------------------------------------------------------
const PLOT_ARCHITECTURES = [
    (arch=:PeNODE, label="PeN-ODE", configs=[54, 83, 55]),
    (arch=:NeuralFMU, label="NeuralFMU", configs=[86, 82, 74]),
]
const NROWS = 3
const PLOT_STEPS = 15000

# color blind friendly color scheme plus a neon mint as fifth color
# (neon/lime greens are nearly indistinguishable from col3 for deuteranopes, mint stays distinct for all CVD types)
for line in [
    raw"\definecolor{col1}{RGB}{216,27,96}",
    raw"\definecolor{col2}{RGB}{30,136,229}",
    raw"\definecolor{col3}{RGB}{255,193,7}",
    raw"\definecolor{col4}{RGB}{0,77,64}",
    raw"\definecolor{col5}{RGB}{29,233,182}",
]
    line in PGFPlotsX.CUSTOM_PREAMBLE || push!(PGFPlotsX.CUSTOM_PREAMBLE, line)
end
const SEED_COLORS = ["col2", "col1", "col3", "col4", "col5"]
# additional line style per seed (moving averages), so seeds are distinguishable without color
const SEED_STYLES = ["solid", "dashed", "densely dotted", "dashdotted", "dashdotdotted"]

# y axis scale, same range for all panels
# log:    0.005-0.5 covers all runs, incl. the spike of config 83, seed 123, up to 0.36
# linear: 0-0.2 clips only that spike
const LOG_Y = false
const YMIN, YMAX = LOG_Y ? (5e-3, 0.5) : (0.0, 0.2)

# centered moving average over SMOOTH_WINDOW validation points (window shrinks at the ends)
const SMOOTH_WINDOW = 11 # = 550 training steps
function moving_average(v, window)
    half = window ÷ 2
    return [mean(v[max(1, i - half):min(length(v), i + half)]) for i in eachindex(v)]
end

function validation_axis(plotarch, row, col)
    config = plotarch.configs[row]
    first_col = col == 1
    last_row = row == NROWS

    # two legends below the grid, both centered: seeds (bottom right panel) and line styles (bottom left panel)
    legend_at = first_col ? "{(1.07,-0.56)}" : "{(-0.07,-0.42)}"
    ax = @pgf Axis({
        xmin = 0, xmax = PLOT_STEPS,
        ymin = YMIN, ymax = YMAX,
        scaled_x_ticks = false,
        xtick = "{0,5000,10000,15000}",
        xticklabel_style = {"/pgf/number format/1000 sep" = "{}"},
        legend_columns = -1, # all entries in one row
        legend_style = {at = legend_at, anchor = "north", draw = "none",
            font = raw"\footnotesize", column_sep = "6pt"},
    })
    if LOG_Y
        ax.options["ymode"] = "log"
    else
        ax.options["scaled y ticks"] = false
        ax.options["yticklabel style"] = @pgf {"/pgf/number format/fixed", "/pgf/number format/precision" = 2}
    end
    # shared axes: tick labels and axis labels only on the outer panels
    row == 1 && (ax.options["title"] = plotarch.label)
    last_row ? (ax.options["xlabel"] = "training step") : (ax.options["xticklabels"] = "{}")
    first_col || (ax.options["yticklabels"] = "{}")
    first_col && row == 2 && (ax.options["ylabel"] = "validation MAE [A]") # once, centered vertically

    push!(ax, "\\node[anchor=north east, font=\\footnotesize] at (rel axis cs:0.98,0.98) {config $(config)};")

    # validation MAE every VALID_INTERVAL steps (without the extra final evaluation)
    losses = map(SEEDS) do seed
        h = load_validation_history(seed_run_dir(RESULTS_DIR, ARCH_DIRS[plotarch.arch], config, seed))
        h.valid_losses[1:(h.laststep ÷ VALID_INTERVAL)]
    end

    # raw validation trajectories: thin and faint in the background
    for (k, v) in enumerate(losses)
        push!(ax, @pgf Plot({no_marks, color = SEED_COLORS[k], line_width = "0.3pt", opacity = 0.3, forget_plot},
            Coordinates(VALID_INTERVAL .* (1:length(v)), v)))
    end
    # moving averages on top
    seed_legend = last_row && col == 2
    for (k, v) in enumerate(losses)
        p = @pgf Plot({no_marks, color = SEED_COLORS[k], line_width = "1pt", forget_plot = !seed_legend},
            Coordinates(VALID_INTERVAL .* (1:length(v)), moving_average(v, SMOOTH_WINDOW)))
        p.options[SEED_STYLES[k]] = nothing # line style as additional, color independent cue
        push!(ax, p)
        seed_legend && push!(ax, LegendEntry("seed $(k)"))
    end
    # meaning of the line styles (bottom left panel, placed below the seed legend)
    if last_row && first_col
        push!(ax, "\\addlegendimage{no marks, color=black!60, line width=1pt}" *
                  "\\addlegendentry{moving average ($(SMOOTH_WINDOW * VALID_INTERVAL) steps)}")
        push!(ax, "\\addlegendimage{no marks, color=black!60, line width=0.3pt, opacity=0.3}" *
                  "\\addlegendentry{validation MAE (every $(VALID_INTERVAL) steps)}")
    end
    return ax
end

gp = @pgf GroupPlot(
    {group_style = {group_size = "2 by $(NROWS)", horizontal_sep = "0.8cm", vertical_sep = "0.4cm"},
        width = "7cm", height = "4.2cm"},
    [validation_axis(PLOT_ARCHITECTURES[col], row, col) for row in 1:NROWS for col in 1:2]...,
)

basename = joinpath(FIG_DIR, "seed_validation_losses")
pgfsave(basename * ".pdf", gp) # requires a LaTeX installation
pgfsave(basename * ".svg", gp)
println("saved: ", basename, ".{pdf,svg}")
println("seeds 1-5 in the figure: ", join(["$(k) = $(s)" for (k, s) in enumerate(SEEDS)], ", "))
