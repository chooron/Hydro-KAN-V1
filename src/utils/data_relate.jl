
using JLD2

function normalize_data(vec)
	mean, std = mean(vec), std(vec)
	norm_func = (x) -> (x .- mean) ./ std
	return norm_func.(vec), norm_func
end


function load_nn_data(basin_id, basemodel_name="exphydro(516)")
    basemodel_dir = "result/v2/$basemodel_name"

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

    norm_snowpack = (snowpack_vec .- s0_mean) ./ s0_std
    norm_soilwater = (soilwater_vec .- s1_mean) ./ s1_std
    norm_temp = (temp_vec .- t_mean) ./ t_std
    norm_infil = (infil_vec .- infil_mean) ./ infil_std
    norm_lday = (lday_vec .- lday_mean) ./ lday_std

    nn_input = stack([norm_snowpack, norm_soilwater, norm_temp, norm_lday, norm_infil], dims=1)
    etnn_input = nn_input[[1, 2, 3, 4], 1:length(train_timepoints)]
    qnn_input = nn_input[[1, 2, 5], 1:length(train_timepoints)]

    return etnn_input, qnn_input
end


function load_nn_data_v2(basin_id, basemodel_name="exphydro(516)")
    basemodel_dir = "result/v2/$basemodel_name"

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

    norm_snowpack = (snowpack_vec .- s0_mean) ./ s0_std
    norm_soilwater = (soilwater_vec .- s1_mean) ./ s1_std
    norm_temp = (temp_vec .- t_mean) ./ t_std
    norm_infil = (infil_vec .- infil_mean) ./ infil_std
    norm_lday = (lday_vec .- lday_mean) ./ lday_std

    nn_input = stack([norm_snowpack, norm_soilwater, norm_temp, norm_lday, norm_infil], dims=1)
    etnn_input = nn_input[[1, 2, 3, 4], 1:length(train_timepoints)]
    qnn_input = nn_input[[1, 2, 5], 1:length(train_timepoints)]

    melt_prop_vec = melt_vec ./ infil_vec
    rainfall_prop_vec = pr_vec ./ infil_vec
    melt_prop_vec[isnan.(melt_prop_vec)] .= 0.0
    rainfall_prop_vec[isnan.(rainfall_prop_vec)] .= 0.0

    return etnn_input, qnn_input, melt_prop_vec, rainfall_prop_vec
end