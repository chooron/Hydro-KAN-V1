using Pkg
Pkg.activate(".")
using Lux
using Plots
using StableRNGs
using Zygote
using Statistics
using Random
using ComponentArrays
using ProgressMeter
using HydroModels
using ParameterSchedulers: Scheduler, Exp, Step
using DifferentialEquations, SciMLSensitivity, DataInterpolations
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using CSV, DataFrames, Dates, DelimitedFiles, JLD2

include("../models/m50.jl")
include("../utils/train.jl")
include("../utils/criteria.jl")
include("../utils/kan_tools.jl")

function main(basin_id, reg_gamma)
    reg_gamma_dict = Dict("1e-2" => 1e-2, "1e-3" => 1e-3, "5e-3" => 5e-3)
    model_name = "k50_reg($reg_gamma)"
    save_model_name = "k50_prune($reg_gamma)"

    basemodel_name = "exphydro(516)"
    basemodel_dir = "result/$basemodel_name"

    #* load data
    camelsus_cache = load("data/camelsus/$(basin_id).jld2")
    data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
    train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
    test_x, test_y, test_timepoints = camelsus_cache["test_x"], camelsus_cache["test_y"], camelsus_cache["test_timepoints"]
    lday_vec, prcp_vec, temp_vec = collect(data_x[:, 1]), collect(data_x[:, 2]), collect(data_x[:, 3])
    # Replace missing values with 0.0 in data arrays
    for data_arr in [data_x, data_y, train_x, train_y, test_x, test_y]
        replace!(data_arr, missing => 0.0)
    end

    #* load parameters and initial states
    exphydro_pas = load("$(basemodel_dir)/$(basin_id)/opt_params.jld2")["opt_params"]
    exphydro_params = NamedTuple{(:f, :Smax, :Qmax, :Df, :Tmax, :Tmin)}(exphydro_pas[1:6]) |> ComponentArray
    exphydro_init_states = NamedTuple{(:snowpack, :soilwater)}(exphydro_pas[7:8]) |> ComponentArray

    #* load exphydro model outputs
    exphydro_df = CSV.read("$(basemodel_dir)/$(basin_id)/model_outputs.csv", DataFrame)
    snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
    et_vec, flow_vec = exphydro_df[!, "et"], exphydro_df[!, "qsim"]
    pr_vec, melt_vec = exphydro_df[!, "pr"], exphydro_df[!, "melt"]
    infil_vec = pr_vec .+ melt_vec
    et_vec[et_vec.<0] .= 0.000000001
    flow_vec[flow_vec.<0] .= 0.000000001
    infil_vec[infil_vec.<0] .= 0.000000001

    #* normalize data
    s0_mean, s0_std = mean(snowpack_vec), std(snowpack_vec)
    s1_mean, s1_std = mean(soilwater_vec), std(soilwater_vec)
    t_mean, t_std = mean(temp_vec), std(temp_vec)
    infil_mean, infil_std = mean(infil_vec), std(infil_vec)
    lday_mean, lday_std = mean(lday_vec), std(lday_vec)

    stand_params = [
        s0_std, s0_mean, s1_std, s1_mean, t_std, t_mean,
        lday_std, lday_mean, infil_std, infil_mean
    ]
    norm_snowpack = (snowpack_vec .- s0_mean) ./ s0_std
    norm_soilwater = (soilwater_vec .- s1_mean) ./ s1_std
    norm_temp = (temp_vec .- t_mean) ./ t_std
    norm_infil = (infil_vec .- infil_mean) ./ infil_std
    norm_lday = (lday_vec .- lday_mean) ./ lday_std

    #* build model
    original_et_nn, original_q_nn = build_nns()
    etnn_ps_axes = getaxes(LuxCore.initialparameters(Random.default_rng(), original_et_nn) |> ComponentArray)
    qnn_ps_axes = getaxes(LuxCore.initialparameters(Random.default_rng(), original_q_nn) |> ComponentArray)
    #* prepare training data
    nn_input = stack([norm_snowpack, norm_soilwater, norm_temp, norm_lday, norm_infil], dims=1)
    etnn_input = nn_input[[1, 2, 3, 4], 1:length(train_timepoints)]
    qnn_input = nn_input[[1, 2, 5], 1:length(train_timepoints)]
    #* prune nodes
    kan_ckpt = load("result/$(model_name)/$(basin_id)/train_records.jld2")["reg_opt_ps"]
    pruned_et_nn_layers, pruned_et_ps, etnn_nodes_to_keep, etnn_node_scores = prune_etnn_nodes(
        model_layers=[original_et_nn.layer_1, original_et_nn.layer_2],
        layer_params=ComponentVector(kan_ckpt["nns"]["etnn"], etnn_ps_axes),
        input_data=etnn_input, prune_threshold=1 / 6,
    )
    pruned_q_nn_layers, pruned_q_ps, qnn_nodes_to_keep, qnn_node_scores = prune_qnn_nodes(
        model_layers=[original_q_nn.layer_1, original_q_nn.layer_2],
        layer_params=ComponentVector(kan_ckpt["nns"]["qnn"], qnn_ps_axes),
        input_data=qnn_input, prune_threshold=1 / 6,
    )
    pruned_et_nn = Lux.Chain(pruned_et_nn_layers...; name=:etnn)
    pruned_q_nn = Lux.Chain(pruned_q_nn_layers...; name=:qnn)

    # etnn, qnn = build_mlp_nns(10)
    m50_model = build_m50_model(pruned_et_nn, pruned_q_nn, stand_params)
    model_init_pas = ComponentArray(
        nns=(etnn=Vector(ComponentVector(pruned_et_ps)), qnn=Vector(ComponentVector(pruned_q_ps))),
        params=exphydro_params
    )
    ps_axes = getaxes(model_init_pas)
    # lday, prcp, temp => prcp, temp, lday
    train_arr = permutedims(train_x)[[2, 3, 1], :]
    test_arr = permutedims(test_x)[[2, 3, 1], :]
    total_arr = permutedims(data_x)[[2, 3, 1], :]

    initstates = ComponentArray(NamedTuple{(:snowpack, :soilwater)}(exphydro_init_states))
    config = (
        solver=HydroModels.DiscreteSolver(),
        interp=DataInterpolations.LinearInterpolation
    )
    #* optimization
    model_func(x, p) = m50_model(x, ComponentVector(p, ps_axes), initstates=initstates, config=config)[end, :]
    train_best_ps, val_best_ps, train_recorder_df = train_hybrid(
        model_func,
        (train_arr, train_y, train_timepoints),
        (total_arr, data_y, data_timepoints),
        model_init_pas,
        loss_func=nse_loss,
        reg_func=reg_loss(reg_gamma_dict[reg_gamma], [:nns]),
        optmzr=ADAM(1e-3), max_N_iter=100, warm_up=1,
        adtype=Optimization.AutoZygote()
    )
    #* re-retrain with best parameters
    retrain_best_ps, retrain_val_best_ps, retrain_recorder_df = train_hybrid(
        model_func,
        (train_arr, train_y, train_timepoints),
        (total_arr, data_y, data_timepoints),
        val_best_ps,
        loss_func=nse_loss,
        reg_func=reg_loss(reg_gamma_dict[reg_gamma], [:nns]),
        optmzr=LBFGS(linesearch=BackTracking()), max_N_iter=100, warm_up=1,
        adtype=Optimization.AutoZygote()
    )
    output = m50_model(train_arr, retrain_best_ps, initstates=initstates, config=config)
    model_out_names = vcat(HydroModels.get_state_names(m50_model), HydroModels.get_output_names(m50_model))
    output_df = DataFrame(NamedTuple{Tuple(model_out_names)}(eachrow(output)))
    # * save calibration results
    # * save calibration results
    model_dir = "result/$(save_model_name)/$(basin_id)"
    mkpath(model_dir)
    save(
        "$(model_dir)/train_records.jld2",
        "opt_ps", train_best_ps, "val_opt_ps", val_best_ps,
        "reg_opt_ps", retrain_best_ps, "reg_val_opt_ps", retrain_val_best_ps,
    )
    save("$(model_dir)/loss_df.jld2",
        "adam_loss_df", train_recorder_df, "lbfgs_loss_df", retrain_recorder_df
    )
    save("$(model_dir)/other_info.jld2",
        "output_df", output_df,
        "etnn_nodes_to_keep", etnn_nodes_to_keep,
        "qnn_nodes_to_keep", qnn_nodes_to_keep,
    )
    #* make prediction
    y_pred = model_func(total_arr, train_best_ps)
    y_val_pred = model_func(total_arr, val_best_ps)
    y_reg_pred = model_func(total_arr, retrain_best_ps)
    y_val_reg_pred = model_func(total_arr, retrain_val_best_ps)
    save(
        "$(model_dir)/predicted_df.jld2",
        "y_pred", y_pred, "y_val_pred", y_val_pred,
        "y_reg_pred", y_reg_pred, "y_val_reg_pred", y_val_reg_pred
    )
    GC.gc(true)
end

# Read criteria file and filter stations with negative NSE
reg_gamma = ARGS[1]
model_name = "k50_reg($reg_gamma)"
basin_file = load("src/cache/$(model_name)_basin_filtered.jld2")["basin_filtered"]
basins_available = lpad.(string.(Int.(basin_file)), 8, "0")
for basin_id in basins_available
    save_model_name = "k50_prune($reg_gamma)"
    if !ispath("result/$(save_model_name)/$(basin_id)")
        println("running basin $basin_id")
        main(basin_id, reg_gamma)
    else
        println("skipping basin $basin_id")
    end
end

# "src/reg/run_k50_prune.jl"