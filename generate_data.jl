#
# Copyright (c) 2026 Andreas Hofmann
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

#
# Generation of the SCARA reference data (training, validation, test) and of the test data
# of the SCARA with modified arm lengths (L1 = 0.25 m, L2 = 0.15 m).
#
# usage (in the repository root): julia --project=. generate_data.jl
#

using ScaraModel
using ScaraModel.GenerateReferenceData
using ModelingToolkit
using DifferentialEquations
using HDF5
using DataInterpolations
using DelimitedFiles

# input trajectories (columns: t [s], x [m], y [m], normal force [N]) and output directory
const INPUT_DIR = joinpath(@__DIR__, "data")
const OUTPUT_DIR = joinpath(@__DIR__, "data")
mkpath(OUTPUT_DIR)


# ---------------------------------------------------------------------------
# reference data of the SCARA with friction (L1 = 0.2 m, L2 = 0.1 m)
# ---------------------------------------------------------------------------
generateReferenceData(joinpath(INPUT_DIR, "train.txt"), "train", joinpath(OUTPUT_DIR, "ScaraTrain.h5"))
generateReferenceData(joinpath(INPUT_DIR, "validate.txt"), "validate", joinpath(OUTPUT_DIR, "ScaraValidate.h5"))
generateReferenceData(joinpath(INPUT_DIR, "thanks.txt"), "test", joinpath(OUTPUT_DIR, "ScaraTest.h5"))


# ---------------------------------------------------------------------------
# reference data of the SCARA with friction and modified arm lengths
# ---------------------------------------------------------------------------

# inputs as in ScaraModel.GenerateReferenceData.generateReferenceData: linear interpolation with linear extrapolation
function getScaraInputs(path_to_data)
    trajs = DelimitedFiles.readdlm(path_to_data)
    return [DataInterpolations.LinearInterpolation(trajs[:, i], trajs[:, 1]; extrapolation=DataInterpolations.ExtrapolationType.Linear)
            for i in 2:4]
end

# same as ScaraModel.GenerateReferenceData.generateReferenceData (model, initial values, solver settings,
# saveat, tstops and HDF5 layout), but with the arm lengths L1, L2 set
function generateReferenceDataScaledLengths(inputdata::String, simname::String, h5file::String; L1, L2, samplingrate=1e-4)

    (inp1, inp2, inp3) = getScaraInputs(inputdata)

    @mtkcompile scara = ScaraModel.ScaraRobot.ScaraMTKModel(inp1=inp1, inp2=inp2, inp3=inp3, L1=L1, L2=L2)

    # result quantities
    state_quantities = unknowns(scara)

    input_quantities = [
        (inp1, "inp1"),
        (inp2, "inp2"),
        (inp3, "inp3")
    ]

    output_quantities = [
        (scara.scara.tcp.vx, "TCP.vx"),
        (scara.scara.tcp.vy, "TCP.vy"),
        (scara.scara.tcp.x, "TCP.x"),
        (scara.scara.tcp.y, "TCP.y"),
        (scara.scara.tcp.fx, "TCP.FrictionForceX"),
        (scara.scara.tcp.fy, "TCP.FrictionForceY")
    ]

    tspan = (inp1.t[1], inp1.t[end])

    u0 = [
        scara.scara.rev1.phi => 0.0,
        scara.scara.rev2.phi => 0.0,
        scara.scara.rev1.w => 0.0,
        scara.scara.rev2.w => 0.0,
        scara.motor1.inductor.i => 0.0,
        scara.motor2.inductor.i => 0.0
    ]

    prob = ODEProblem(scara, u0, tspan)

    sol = solve(prob, Rodas5P(), saveat=samplingrate, tstops=inp1.t)

    HDF5.h5open(h5file, "w") do file
        gr = HDF5.create_group(file, simname)

        for (i, entry) in enumerate(state_quantities)
            item_group = HDF5.create_group(gr, string(entry))
            item_group["value"] = sol(sol.t, Val{0}; idxs=i).u
            item_group["time"] = sol.t
            der_group = HDF5.create_group(gr, "der_" * string(entry))
            value = sol(sol.t, Val{1}; idxs=i).u
            der_group["value"] = value
            der_group["time"] = sol.t
        end

        for entry in input_quantities
            item_group = HDF5.create_group(gr, entry[2])
            item_group["value"] = entry[1].u
            item_group["time"] = entry[1].t
        end

        for entry in output_quantities
            item_group = HDF5.create_group(gr, entry[2])
            item_group["value"] = sol[entry[1]]
            item_group["time"] = sol.t
        end
    end

    return string.(state_quantities)
end

# test trajectory with longer arms: L1 = 0.25 m, L2 = 0.15 m
generateReferenceDataScaledLengths(joinpath(INPUT_DIR, "thanks.txt"), "test", joinpath(OUTPUT_DIR, "ScaraTest_L1_025_L2_015.h5");
    L1=0.25, L2=0.15)
