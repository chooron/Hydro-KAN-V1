using Pkg
Pkg.activate(".")

using DifferentialEquations
using Statistics
using ComponentArrays
using DataInterpolations
using ProgressMeter
using Optimization, OptimizationBBO
using DelimitedFiles
using SciMLSensitivity
using CSV, DataFrames, Dates, JLD2

include("../models/exphydro.jl")
include("../utils/HydroNODE_data.jl")

input_var_names = ["Daylight(h)", "Prec(mm/day)", "Tmean(C)"]
output_var_name = "Flow(mm/s)"
data_path = "G:\\Dataset\\CAMELS_US"
model_name = "exphydro(0605)" # 离散模型,考虑初始状态
nse_loss(obs, pred) = sum((pred .- obs) .^ 2) / sum((obs .- mean(obs)) .^ 2)

function calibrate_exphydro(func, data, pas;
    lb, ub,
    loss_func=nse_loss, warmup=1,
    optmzr=BBO_adaptive_de_rand_1_bin_radiuslimited(), max_N_iter=10000
)
    x, y = data
    progress = Progress(max_N_iter, desc="Training...")
    recorder = []

    function objective(u, p)
        y_hat = func(x, u)
        return loss_func(y[warmup:length(y)], y_hat[warmup:length(y_hat)])
    end

    function callback(state, l)
        push!(recorder, (iter=state.iter, loss=l, time=now()))
        next!(progress)
        return false
    end

    optf = Optimization.OptimizationFunction(objective)
    optprob = Optimization.OptimizationProblem(optf, pas, lb=lb, ub=ub)
    sol = Optimization.solve(optprob, optmzr, maxiters=max_N_iter, callback=callback)
    recorder_df = DataFrame(recorder)
    return sol.u, recorder_df
end

function main(basin_id)
    camelsus_cache = load("data/camelsus/$(basin_id).jld2")
    data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
    train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
    test_x, test_y, test_timepoints = camelsus_cache["test_x"], camelsus_cache["test_y"], camelsus_cache["test_timepoints"]
    # lday_vec, prcp_vec, temp_vec = collect(data_x[:, 1]), collect(data_x[:, 2]), collect(data_x[:, 3])
    # Inputs: [prcp, temp, lday]
    train_arr = permutedims(train_x)[[2, 3, 1], :]
    total_arr = permutedims(data_x)[[2, 3, 1], :]
    config = (solver=HydroModels.ODESolver(), interp=LinearInterpolation,)

    lower_bounds = [0.0, 100.0, 10.0, 0.01, 0.0, -5.0, 0.01, 100.0]
    upper_bounds = [0.5, 1500.0, 100.0, 10.0, 5.0, 0.0, 1500.0, 1500.0]
    default_ps_vec = lower_bounds .+ 0.5 * (upper_bounds .- lower_bounds)
    default_params = ComponentArray(NamedTuple{(:f, :Smax, :Qmax, :Df, :Tmax, :Tmin)}(default_ps_vec[1:6]))
    default_initstates = ComponentArray(NamedTuple{(:snowpack, :soilwater)}(default_ps_vec[7:8]))
    default_pas = ComponentArray(params=default_params, initstates=default_initstates)
    default_pas_axes = getaxes(default_pas)
    model_func(x, p) = begin
        p_cv = ComponentArray(p, default_pas_axes)
        exphydro_model(x, p_cv, initstates=p_cv.initstates, config=config)[end, :]
    end
    opt_params, opt_recorder = calibrate_exphydro(
        model_func, (train_arr, train_y), Vector(default_pas),
        lb=lower_bounds, ub=upper_bounds,
        max_N_iter=10000, warmup=1, loss_func=nse_loss
    )

    opt_params = ComponentArray(opt_params, default_pas_axes)
    output = exphydro_model(total_arr, opt_params, initstates=opt_params.initstates, config=config)
    model_output_names = vcat(HydroModels.get_state_names(exphydro_model), HydroModels.get_output_names(exphydro_model))
    output_df = DataFrame(NamedTuple{Tuple(model_output_names)}(eachslice(output, dims=1)))
    save_path = "result/v2/$(model_name)/$(basin_id)"
    mkpath(save_path)
    CSV.write("$(save_path)/model_outputs.csv", output_df)
    CSV.write("$(save_path)/opt_recorder.csv", opt_recorder)
    train_pred_df = DataFrame((pred=output[end, 1:length(train_timepoints)], obs=train_y))
    test_pred_df = DataFrame((pred=output[end, length(train_timepoints)+1:end], obs=test_y))
    CSV.write("$(save_path)/train_predicted_df.csv", train_pred_df)
    CSV.write("$(save_path)/test_predicted_df.csv", test_pred_df)
    save("$(save_path)/opt_params.jld2", "opt_params", opt_params)
end

basins_id_file = readdlm("data/basin_ids/basins_all.txt")
basins_available = lpad.(string.(Int.(basins_id_file)), 8, "0")
reverse_basins = ARGS[1]
if reverse_basins == "1"
    println("reversing basins")
    basins_available = reverse(basins_available)
end
for basin_id in basins_available
    save_path = "result/v2/$(model_name)/$(basin_id)"
    if !isdir(save_path)
        main(basin_id)
    else
        println("$(basin_id) already exists")
    end
end