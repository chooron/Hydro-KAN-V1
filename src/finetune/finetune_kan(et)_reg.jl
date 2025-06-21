# ! finetune kan(et) with different hidden dimensions and grid sizes
using Pkg
Pkg.activate(".")
using Lux
using Plots
using StableRNGs
using Zygote
using Statistics
using ComponentArrays
using HydroModels
using KolmogorovArnold
using Optimization, OptimizationOptimisers
using CSV, DataFrames, Dates, JLD2, JSON, DelimitedFiles
# using CUDA, cuDNN
using Lux, LuxCUDA
# CUDA.allowscalar(false)
# device = Lux.gpu_device()
device = Lux.cpu_device()
model_name = "kan(et)"
include("../utils/train.jl")
include("../utils/criteria.jl")
include("../utils/kan_tools.jl")

function finetune_kan(basin_id, reg_gamma)
    model_name = "kan(et)"
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

    #* load exphydro model outputs
    exphydro_df = CSV.read("$(basemodel_dir)/$(basin_id)/model_outputs.csv", DataFrame)
    snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
    et_vec = exphydro_df[!, "et"]
    et_vec[et_vec.<0] .= 0.000000001

    #* normalize data
    s0_mean, s0_std = mean(snowpack_vec), std(snowpack_vec)
    s1_mean, s1_std = mean(soilwater_vec), std(soilwater_vec)
    temp_mean, temp_std = mean(temp_vec), std(temp_vec)
    lday_mean, lday_std = mean(lday_vec), std(lday_vec)

    norm_snowpack = (snowpack_vec .- s0_mean) ./ s0_std
    norm_soilwater = (soilwater_vec .- s1_mean) ./ s1_std
    norm_lday = (lday_vec .- lday_mean) ./ lday_std
    norm_temp = (temp_vec .- temp_mean) ./ temp_std

    #* build model
    etnn = Chain(
        KDense(4, 3, 6; use_base_act=true),
        (x) -> prod(x, dims=1),
        name=:etnn,
    )
    et_nn_ps, et_nn_st = Lux.setup(StableRNG(42), etnn)
    et_nn_ps = ComponentVector(et_nn_ps) .|> Float64 |> device
    et_nn_st = device(et_nn_st)
    et_apply = (x, p) -> etnn(x, p, et_nn_st)[1]
    #* prepare training data
    etnn_input = stack([norm_snowpack, norm_soilwater, norm_temp, norm_lday], dims=1) |> device
    etnn_target = reshape(et_vec, 1, :) |> device
    #* train NNs
    opt_et_ps, etnn_loss_recorder = train_nn_v2(
        et_apply, (etnn_input[:, 1:length(train_timepoints)], etnn_target[:, 1:length(train_timepoints)]),
        (etnn_input[:, length(train_timepoints)+1:end], etnn_target[:, length(train_timepoints)+1:end]),
        et_nn_ps, optmzr=ADAM(0.01), max_N_iter=2000,
        reg_func=reg_loss(reg_gamma, [:layer_1])
    )
    train_pred = et_apply(etnn_input[:, 1:length(train_timepoints)], opt_et_ps)
    test_pred = et_apply(etnn_input[:, length(train_timepoints)+1:end], opt_et_ps)
    train_mse = mse_loss(train_pred[1, :], et_vec[1:length(train_timepoints)] |> device)
    test_mse = mse_loss(test_pred[1, :], et_vec[length(train_timepoints)+1:end] |> device)
    mkpath("result/finetune/$(model_name)_$(reg_gamma)")
    mkpath("result/finetune/$(model_name)_$(reg_gamma)/spline")
    mkpath("result/finetune/$(model_name)_$(reg_gamma)/params")
    spline_dict = obtain_etnn_splines(model_layers=[etnn.layer_1], layer_params=opt_et_ps, input_data=etnn_input[:, length(train_timepoints)+1:end])
    save("result/finetune/$(model_name)_$(reg_gamma)/spline/$(basin_id).jld2", "postacts1", spline_dict["postacts1"])
    save("result/finetune/$(model_name)_$(reg_gamma)/params/$(basin_id).jld2", "opt_ps", opt_et_ps)
    open("result/finetune/$(model_name)_$(reg_gamma)/$(basin_id).json", "w") do f
        JSON.print(f, Dict(
            "train_mse" => train_mse, "test_mse" => test_mse,
            "init_reg_loss" => etnn_loss_recorder[1, :reg_loss] / reg_gamma,
            "final_reg_loss" => etnn_loss_recorder[end, :reg_loss] / reg_gamma
        ))
    end
    GC.gc(true)
end

reverse_basins = ARGS[1]
basin_file = readdlm(joinpath("data/basin_ids/basins_all.txt"), ',')
basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")
if reverse_basins == "1"
    println("reversing basins")
    basins_available = reverse(basins_available)
end

for basin_id in basins_available
    for reg_gamma in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        if !ispath("result/finetune/$(model_name)_$(reg_gamma)/$(basin_id).json")
            println("running basin $basin_id")
            finetune_kan(basin_id, reg_gamma)
        else
            println("skipping basin $basin_id at $reg_gamma")
        end
    end
end
