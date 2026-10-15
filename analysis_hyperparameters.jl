#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#
# Analysis of the hyperparameter study (100 configurations per architecture, last checkpoint of every run)
#   1. validation and test MAE of every run (cached in hyperparameter_losses.csv)
#   2. Wilcoxon signed-rank test PeN-ODE vs. NeuralFMU (paired by configuration)
#   3. correlation of validation and test MAE (Spearman, Kendall, top-k overlap)
#   4. figures: paired scatter plot of the validation MAE, dumbbell plots of validation vs. test MAE
#
# usage (in the repository root): julia --project=. analysis_hyperparameters.jl
# requires the reference data in data/ (see generate_data.jl) when the losses are (re)computed
#

using MiniPeNODETrainingFramework
using Statistics
using HypothesisTests
using CSV
using PGFPlotsX
using PGFPlotsX: Plot, Axis, Coordinates, Legend, LegendEntry # explicit, PlotlyJS (loaded via MiniPeNODETrainingFramework) also exports Plot

include("util/search_space.jl")
include("util/penode_postprocess.jl")
include("util/neuralfmu_postprocess.jl")
include("util/correlationTest.jl")

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
# checkpoints of the delivered hyperparameter study; for a new training with train.jl (STUDY = :hyperparameters)
# use RESULTS_DIR = @__DIR__ and the PeN-ODE directory "OOM"
const RESULTS_DIR = joinpath(@__DIR__, "hyperparameter_results")
const ARCH_DIRS = (PeNODE="PeNODE", NeuralFMU="NeuralFMU")
const CHECKPOINT = 5000
const NRUNS = 100

const LOSS_FILE = joinpath(@__DIR__, "hyperparameter_losses.csv")
const RECOMPUTE = false # true: evaluate all checkpoints again (two simulations per run, takes a while)
const TOPK = (5, 10)

const FIG_DIR = joinpath(@__DIR__, "figures")
mkpath(FIG_DIR)

const ACT_NAMES = Dict(tanh => "tanh", relu => "relu", sigmoid => "sigmoid")

# ---------------------------------------------------------------------------
# 1. validation and test MAE of every run
# ---------------------------------------------------------------------------
function evaluate_hyperparameter_runs()
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
    for n in 1:NRUNS
        println("evaluating run $(n)/$(NRUNS)")
        args = (input_signals, meas_model_mapping, loss_selection, combs[n], n, training_data, validation_data, test_data)
        (valid_penode, test_penode) = simulateTrainedPeNODE(args...; parameterid=CHECKPOINT,
            customdir=joinpath(RESULTS_DIR, ARCH_DIRS.PeNODE, "run_$(n)"))
        (valid_neuralfmu, test_neuralfmu) = simulateTrainedNeuralFMU(args...; parameterid=CHECKPOINT,
            customdir=joinpath(RESULTS_DIR, ARCH_DIRS.NeuralFMU, "run_$(n)"))
        push!(rows, (run=n, num_neurons=combs[n][1], num_hidden=combs[n][2], type_act=ACT_NAMES[combs[n][3]],
            valid_penode=valid_penode, test_penode=test_penode, valid_neuralfmu=valid_neuralfmu, test_neuralfmu=test_neuralfmu))
    end
    CSV.write(LOSS_FILE, identity.(rows))
end

(RECOMPUTE || !isfile(LOSS_FILE)) && evaluate_hyperparameter_runs()
losses = CSV.File(LOSS_FILE)
@assert collect(losses.run) == 1:NRUNS "runs in $(LOSS_FILE) are not 1:$(NRUNS)"

valid_penode = collect(losses.valid_penode)
test_penode = collect(losses.test_penode)
valid_neuralfmu = collect(losses.valid_neuralfmu)
test_neuralfmu = collect(losses.test_neuralfmu)

# ---------------------------------------------------------------------------
# 2. Wilcoxon signed-rank test: PeN-ODE vs. NeuralFMU, paired by configuration
# ---------------------------------------------------------------------------
wilcoxon = SignedRankTest(valid_neuralfmu, valid_penode)
println("\n=== Wilcoxon signed-rank test: validation MAE NeuralFMU vs. PeN-ODE ($(NRUNS) paired configurations) ===")
println(wilcoxon)
println("\n--- Descriptive statistics of the validation MAE ---")
for (name, v) in (("PeN-ODE", valid_penode), ("NeuralFMU", valid_neuralfmu))
    println("  ", rpad(name, 10), " mean $(round(mean(v), digits=4)), median $(round(median(v), digits=4)), min $(round(minimum(v), digits=4)) (run $(argmin(v)))")
end
println("  PeN-ODE better in $(count(valid_penode .< valid_neuralfmu)) of $(NRUNS) configurations")

# ---------------------------------------------------------------------------
# 3. correlation of validation and test MAE (is the validation MAE a good proxy for the test MAE?)
# ---------------------------------------------------------------------------
correlations = Dict{String,Any}()
for (name, v, t) in (("PeN-ODE", valid_penode, test_penode), ("NeuralFMU", valid_neuralfmu, test_neuralfmu))
    st = SpearmanCorrelationTest(v, t)
    kt = KendallTauTest(v, t)
    tk = Dict(k => TopKOverlapTest(v, t, k) for k in TOPK)
    correlations[name] = (spearman=st, kendall=kt, topk=tk)

    println("\n=== validation vs. test MAE: $(name) ===")
    println("  Spearman rho = $(round(st.r, digits=4)) (p = $(pvalue(st)))")
    println("  Kendall tau  = $(round(kt.τ, digits=4)) (p = $(pvalue(kt)))")
    for k in TOPK
        println("  top-$(k) overlap = $(tk[k].m)/$(k) (p = $(pvalue(tk[k]))), common runs $(tk[k].common)")
    end
    println("  best validation run $(argmin(v)): test rank $(findfirst(==(argmin(v)), sortperm(t))), best test run $(argmin(t))")
end

# ---------------------------------------------------------------------------
# 4. figures
# ---------------------------------------------------------------------------
# color blind friendly color scheme
for line in [
    raw"\definecolor{col1}{RGB}{216,27,96}",
    raw"\definecolor{col2}{RGB}{30,136,229}",
    raw"\definecolor{col3}{RGB}{255,193,7}",
    raw"\definecolor{col4}{RGB}{0,77,64}",
    # open five-pointed star (TikZ only ships the asterisk-like `star`)
    raw"\pgfdeclareplotmark{openstar}{\pgfpathmoveto{\pgfpointpolar{90}{\pgfplotmarksize}}\pgfpathlineto{\pgfpointpolar{126}{0.4\pgfplotmarksize}}\pgfpathlineto{\pgfpointpolar{162}{\pgfplotmarksize}}\pgfpathlineto{\pgfpointpolar{198}{0.4\pgfplotmarksize}}\pgfpathlineto{\pgfpointpolar{234}{\pgfplotmarksize}}\pgfpathlineto{\pgfpointpolar{270}{0.4\pgfplotmarksize}}\pgfpathlineto{\pgfpointpolar{306}{\pgfplotmarksize}}\pgfpathlineto{\pgfpointpolar{342}{0.4\pgfplotmarksize}}\pgfpathlineto{\pgfpointpolar{18}{\pgfplotmarksize}}\pgfpathlineto{\pgfpointpolar{54}{0.4\pgfplotmarksize}}\pgfpathclose\pgfusepathqstroke}",
]
    line in PGFPlotsX.CUSTOM_PREAMBLE || push!(PGFPlotsX.CUSTOM_PREAMBLE, line)
end

function save_figure(fig, name)
    basename = joinpath(FIG_DIR, name)
    pgfsave(basename * ".pdf", fig) # requires a LaTeX installation
    pgfsave(basename * ".svg", fig)
    pdftocairo = Sys.which("pdftocairo") # EPS for the paper, if poppler is installed
    pdftocairo === nothing || run(`$pdftocairo -eps $(basename * ".pdf") $(basename * ".eps")`)
    println("saved: ", basename, ".{pdf,svg", pdftocairo === nothing ? "" : ",eps", "}")
end

# --- paired scatter plot of the validation MAE: color = neurons per layer, size = hidden layers, shape = activation
const ACT_MARKS = ["relu" => "*", "tanh" => "square*", "sigmoid" => "triangle*"]
const ACT_LABELS = Dict("relu" => "ReLU", "tanh" => "tanh", "sigmoid" => "sigmoid")
const DEPTH_SIZE = Dict(1 => 1.4, 2 => 2.1, 3 => 2.8)  # mark size [pt] per number of hidden layers
const WIDTH_META = Dict(8 => 0.5, 16 => 1.5, 32 => 2.5) # colormap slot per neurons per layer

function plot_paired_scatter(a, b, type_act, num_hidden, num_neurons; lims=(1e-2, 1.0))
    n = length(a)
    wins_a, wins_b = count(a .< b), count(b .< a)
    meta = [WIDTH_META[w] for w in num_neurons]
    lo, hi = lims

    ax = @pgf Axis({
        xmode = "log", ymode = "log",
        xmin = lo, xmax = hi, ymin = lo, ymax = hi,
        width = "4.6cm", height = "4.6cm", scale_only_axis,
        axis_lines = "left",
        log_ticks_with_fixed_point,
        xlabel = "PeN-ODE validation MAE [A]",
        ylabel = "NeuralFMU validation MAE [A]",
        # three discrete levels: 8 = blue (col2), 16 = green (col4), 32 = yellow (col3)
        colormap = "{width}{rgb255(0)=(30,136,229) rgb255(99)=(30,136,229) rgb255(101)=(0,77,64) " *
                   "rgb255(199)=(0,77,64) rgb255(201)=(255,193,7) rgb255(300)=(255,193,7)}",
        point_meta_min = 0, point_meta_max = 3,
        colorbar,
        colorbar_style = {at = "{(parent axis.north east)}", anchor = "north west", xshift = "0.3cm",
            width = "0.15cm", ytick = "{0.5,1.5,2.5}", yticklabels = "{8,16,32}",
            ylabel = "neurons per layer", ylabel_style = {font = raw"\footnotesize"},
            yticklabel_style = {font = raw"\footnotesize"}},
        legend_style = {at = "{(1.40,1)}", anchor = "north west", font = raw"\footnotesize",
            legend_cell_align = "left", inner_xsep = "4pt", column_sep = "3pt"},
    })

    # region where the PeN-ODE is better, diagonal and win counts
    push!(ax, "\\fill[black, opacity=0.06] (axis cs:$lo,$lo) -- (axis cs:$hi,$hi) -- (axis cs:$lo,$hi) -- cycle;")
    push!(ax, @pgf Plot({dashed, color = "black!70", forget_plot}, Coordinates([lo, hi], [lo, hi])))
    push!(ax, "\\node[anchor=north west, font=\\footnotesize] at (axis cs:$(1.1lo),$(0.97hi)) {PeN-ODE better: $wins_a/$n};")
    push!(ax, "\\node[anchor=south east, font=\\footnotesize] at (axis cs:$(0.97hi),$(1.1lo)) {NeuralFMU better: $wins_b/$n};")

    for (act, mark) in ACT_MARKS, depth in 1:3
        idx = findall((type_act .== act) .& (num_hidden .== depth))
        isempty(idx) && continue
        push!(ax, @pgf Plot({
                scatter, only_marks, forget_plot, mark = mark, mark_size = DEPTH_SIZE[depth],
                scatter_src = "explicit",
                "scatter/use mapped color" = {draw = "black!80", fill = "mapped color", fill_opacity = 0.85},
            }, Coordinates(a[idx], b[idx]; meta=meta[idx])))
    end

    # configurations with the best validation MAE of each architecture
    i_a, i_b = argmin(a), argmin(b)
    push!(ax, @pgf Plot({only_marks, mark = "openstar", mark_size = 4.5, color = "col1", line_width = "0.9pt"},
        Coordinates([a[i_a]], [b[i_a]])))
    push!(ax, LegendEntry("Run $(i_a) (best PeN-ODE val. loss)"))
    push!(ax, @pgf Plot({only_marks, mark = "diamond", mark_size = 3.5, color = "black", line_width = "0.9pt"},
        Coordinates([a[i_b]], [b[i_b]])))
    push!(ax, LegendEntry("Run $(i_b) (best NeuralFMU val. loss)"))

    # legend keys for the marker encodings
    push!(ax, raw"\addlegendimage{empty legend}\addlegendentry{}") # empty line as separator
    push!(ax, raw"\addlegendimage{empty legend}\addlegendentry{\textit{hidden activations:}}")
    for (act, mark) in ACT_MARKS
        push!(ax, "\\addlegendimage{only marks, mark=$mark, mark size=2.1, draw=black!80, fill=black!40}\\addlegendentry{$(ACT_LABELS[act])}")
    end
    push!(ax, raw"\addlegendimage{empty legend}\addlegendentry{}") # empty line as separator
    push!(ax, raw"\addlegendimage{empty legend}\addlegendentry{\textit{hidden layers:}}")
    for depth in 1:3
        # show all three activation shapes at the size of this depth
        style = "only marks, mark size=$(DEPTH_SIZE[depth]), draw=black!80, fill=black!40"
        image_code = join(("\\draw[$style, mark=$mark] plot coordinates {($(0.3(k - 1))cm,0cm)};"
                           for (k, (_, mark)) in enumerate(ACT_MARKS)), " ")
        image_code *= " \\path (-0.15cm,0cm);" # keep the largest marker clear of the legend border
        push!(ax, "\\addlegendimage{legend image code/.code={$image_code}}\\addlegendentry{$depth}")
    end
    return ax
end

save_figure(plot_paired_scatter(valid_penode, valid_neuralfmu, collect(losses.type_act), collect(losses.num_hidden),
        collect(losses.num_neurons)), "paired_validation_mae")

# --- dumbbell plot: configurations ranked by validation MAE, connector to the test MAE of the same configuration
function plot_dumbbell(valid, test; k=10)
    n = length(valid)
    order = sortperm(valid)
    valloss = valid[order]
    testloss = test[order]
    ranks = 1:n
    overlap = TopKOverlapTest(valid, test, k).m

    connectors = [
        @pgf Plot({no_marks, color = testloss[r] > valloss[r] ? "col1" : "col3", # test worse: red, better: yellow
                line_width = "0.6pt", forget_plot},
            Coordinates([r, r], [valloss[r], testloss[r]]))
        for r in ranks
    ]

    topk_box = """
    \\fill[gray!18, on layer = axis background] ({axis cs:0.5,1} |- {rel axis cs:0,0.004}) rectangle ({axis cs:$(k + 0.5),1} |- {rel axis cs:0,0.996});
    \\node[align=center, font=\\footnotesize, anchor=north]
        at ({axis cs:$((k + 1) / 2),1} |- {rel axis cs:0,0.98}) {top $(k) (val)\\\\overlap $(overlap)/$(k)};
    """

    return @pgf Axis(
        {
            xlabel = "configuration, ranked by validation MAE",
            ylabel = "MAE [A]",
            set_layers,
            ymode = "log",
            xmin = 0, xmax = n + 1,
            xtick_distance = 20,
            tick_align = "outside",
            tick_pos = "left",
            legend_pos = "south east",
            legend_cell_align = "left",
            legend_style = {draw = "none", fill = "none", font = raw"\footnotesize"},
            width = "18cm", height = "7.5cm",
        },
        topk_box,
        connectors...,
        Plot({only_marks, mark = "*", color = "black", mark_size = "1.6pt"},
            Coordinates(ranks, valloss)),
        Plot({only_marks, mark = "*", mark_size = "1.6pt",
                mark_options = {draw = "col2", fill = "white", line_width = "0.6pt"}},
            Coordinates(ranks, testloss)),
        Legend(["validation MAE", "test MAE"]),
    )
end

save_figure(plot_dumbbell(valid_penode, test_penode), "dumbbell_PeNODE")
save_figure(plot_dumbbell(valid_neuralfmu, test_neuralfmu), "dumbbell_NeuralFMU")
