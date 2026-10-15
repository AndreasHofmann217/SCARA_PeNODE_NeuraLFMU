# SCARA_PeNODE_NeuralFMU
Code for the Paper: Object-Oriented PeN-ODEs vs. NeuralFMUs: A Systematic Evaluation of Hybrid Modeling Paradigms in a Modelica-Based Workflow.

## Data, training and analysis

All scripts are run in this directory with the environment defined by `Project.toml`/`Manifest.toml`
(Julia 1.11.6; the packages `MiniPeNODETrainingFramework` and `ScaraModel` are installed from GitHub at the
versions pinned in `Manifest.toml`). Install the environment once:

```
cd SCARA_PeNODE_NeuralFMU
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

and run the scripts with

```
julia --project=. <script>.jl
```

A LaTeX installation is required for the PDF figures (PGFPlotsX).

## 1. Generate the reference data

```
julia --project=. generate_data.jl
```

Simulates the SCARA with friction for the trajectories in `data/` (`train.txt`, `validate.txt`, `thanks.txt`) and writes to `data/`:

| file | content |
|---|---|
| `ScaraTrain.h5`, `ScaraValidate.h5`, `ScaraTest.h5` | training, validation and test data (L1 = 0.2 m, L2 = 0.1 m) |
| `ScaraTest_L1_025_L2_015.h5` | test data with modified arm lengths (L1 = 0.25 m, L2 = 0.15 m) |

## 2. Training (optional)

```
julia --project=. train.jl
```

The trained models of the paper are included (`hyperparameter_results/`, `seed_study_results/`), so this step is only needed to repeat the training. It was run on a compute server and takes long; set `STUDY` in the script:

- `:hyperparameters` – runs 1–100 of the shuffled search space (`util/search_space.jl`) for both architectures, 5000 steps each
- `:seeds` – PeN-ODE configurations 54, 55, 83 and NeuralFMU configurations 74, 82, 86 with seeds 1, 12, 123, 1234, 12345, 15000 steps each

Runs are trained in parallel worker processes (`TOTAL_CORES`, `CORES_PER_WORKER`). Checkpoints are written every 50 steps to `<architecture>/run_<id>[_<seed>]/` in this directory (the PeN-ODE architecture is called `OOM` in the code). A repeated training is not bit-identical to the delivered results, since the training samples are drawn from the global random number generator.

## 3. Analysis

```
julia --project=. analysis_hyperparameters.jl
julia --project=. analysis_seeds.jl
julia --project=. analysis_structural_adaptation.jl
```

| script | content |
|---|---|
| `analysis_hyperparameters.jl` | validation/test MAE of all 100 configurations (last checkpoint), Wilcoxon signed-rank test PeN-ODE vs. NeuralFMU, correlation of validation and test MAE (Spearman, Kendall, top-k overlap), paired scatter and dumbbell plots |
| `analysis_seeds.jl` | best checkpoint (validation MAE) and test MAE of every seed run, summary per configuration, plot of the validation MAE over training |
| `analysis_structural_adaptation.jl` | test MAE of the best PeN-ODE and NeuralFMU run with original and modified arm lengths, export of currents, TCP position and friction force to `structural_adaptation_trajectories.jld2` |

The losses of the first two scripts are cached (`hyperparameter_losses.csv`, `seed_losses.csv`); set `RECOMPUTE = true` to evaluate the checkpoints again. Figures are saved to `figures/`. By default the delivered results are analysed; to analyse a new training, set `RESULTS_DIR = @__DIR__` and the PeN-ODE directory in `ARCH_DIRS` to `"OOM"`.

## Folder structure

```
SCARA_PeNODE_NeuralFMU/
├── Project.toml, Manifest.toml  Julia environment
├── generate_data.jl, train.jl, analysis_*.jl
├── data/                    trajectories (txt) and generated reference data (h5)
├── ANN_Architectures/       ANN of the PeN-ODE (OOM_Architecture.jl) and NeuralFMU
├── models/                  generated SCARA models of both architectures
├── util/                    training (setup_train.jl), search space, evaluation and test helpers
├── hyperparameter_results/  <architecture>/run_<id>/parameters5000.jld2
└── seed_study_results/      <architecture>/run_<config>_<seed>/ (losses.jld2, best checkpoint w.r.t. the
                             validation MAE and last checkpoint parameters15000.jld2)
```

Note: in the delivered checkpoints the entry `losses` (training loss per step) starts with as many zeros as the run had steps (fixed in the current `util/setup_train.jl`); the actual training losses are `losses[number_of_steps+1:end]`.
