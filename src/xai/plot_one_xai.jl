# 使用ExplainableAI对模型的
using Lux
using Zygote
using JLD2, CSV, DataFrames, DelimitedFiles
using ExplainableAI
using StableRNGs
using Statistics
using Plots
using Peaks

include("../models/nns.jl")
include("../models/m50.jl")
include("../utils/data_relate.jl")

function plot_one_xai_scatter_k50!(fig, basin_id, var_type, model_name)
    etnn, qnn = build_nns()
    qnn_train_params = load("result/$(model_name)/$basin_id/train_records.jld2")["reg_opt_ps"][:nns][:qnn]
    output_df = load("result/$(model_name)/$basin_id/other_info.jld2")["output_df"]
    qnn_input = output_df[365:end, [:norm_snw, :norm_slw, :norm_infil]] |> Array |> permutedims

    flow_vec = output_df[365:end, [:flow]] |> Array |> vec
    indices, _ = findmaxima(flow_vec)
    qnn_ps, qnn_st = Lux.setup(StableRNG(42), qnn)
    qnn_ps_axes = getaxes(ComponentArray(qnn_ps))
    qnn_func(x) = qnn(x, ComponentArray(qnn_train_params, qnn_ps_axes), qnn_st)[1]

    analyzer1 = IntegratedGradients(qnn_func)
    expl1 = analyze(qnn_input[:, indices], analyzer1)

    melt_vec = (output_df[365:end, [:melt]]|>Array|>vec)[indices]
    rainfall_vec = (output_df[365:end, [:rainfall]]|>Array|>vec)[indices]
    melt_prop_vec = melt_vec ./ (melt_vec .+ rainfall_vec)
    rainfall_prop_vec = rainfall_vec ./ (melt_vec .+ rainfall_vec)
    melt_prop_vec[isnan.(melt_prop_vec)] .= 0.0
    rainfall_prop_vec[isnan.(rainfall_prop_vec)] .= 0.0

    melt_contrib1 = melt_prop_vec .* expl1.val[3, :] .+ expl1.val[1, :]
    # melt_contrib1 = expl1.val[1, :]
    rainfall_contrib1 = rainfall_prop_vec .* expl1.val[3, :]
    s1_contrib1 = expl1.val[2, :]

    if var_type == "s1"
        x = qnn_input[2, indices]
        y = s1_contrib1

    elseif var_type == "melt"
        x = qnn_input[1, indices]
        y = melt_contrib1
    elseif var_type == "rainfall"
        x = rainfall_prop_vec .* qnn_input[3, indices]
        y = rainfall_contrib1
    end
    return scatter!(
        fig, x, y, alpha=0.5,
        tickfontsize=14, leftmargin=4mm, markerstrokewidth=1.0,
        markerstrokecolor=:black
    )
end

function plot_one_xai_scatter_m50!(fig, basin_id, var_type)
    basemodel_name = "exphydro(516)"
    @info "Loading data for basin $basin_id"
    et_input, qnn_input, prop_vec = load_nn_data_v2(basin_id, basemodel_name)
    # etnn, qnn = build_nns()
    qnn = Lux.Chain(
        Lux.Dense(3, 16, tanh),
        Lux.Dense(16, 16, leakyrelu),
        Lux.Dense(16, 1, leakyrelu),
    )
    qnn_train_params = load("result/models/m50_base/$basin_id/train_records.jld2")["opt_ps"][:q]
    qnn_ps, qnn_st = Lux.setup(StableRNG(42), qnn)
    qnn_ps_axes = getaxes(ComponentArray(qnn_ps))
    qnn_func(x) = qnn(x, ComponentArray(qnn_train_params, qnn_ps_axes), qnn_st)[1]

    qnn_input = qnn_input[:, 365:end]
    qnn_output = qnn_func(qnn_input) |> vec
    indices, _ = findmaxima(qnn_output)

    analyzer1 = IntegratedGradients(qnn_func)
    expl1 = analyze(qnn_input[:, indices], analyzer1)

    melt_prop_vec, rainfall_prop_vec = prop_vec
    melt_contrib1 = melt_prop_vec .* expl1.val[3, :] .+ expl1.val[1, :]
    # melt_contrib1 = expl1.val[1, :]
    rainfall_contrib1 = rainfall_prop_vec .* expl1.val[3, :]
    s1_contrib1 = expl1.val[2, :]

    if var_type == "s1"
        x = qnn_input[2, indices]
        y = s1_contrib1
    elseif var_type == "melt"
        x = qnn_input[1, indices]
        y = melt_contrib1
    elseif var_type == "rainfall"
        x = rainfall_prop_vec .* qnn_input[3, indices]
        y = rainfall_contrib1
    end
    return scatter!(fig, x, y, alpha=0.5,
        tickfontsize=14, leftmargin=4mm, markerstrokewidth=1.0,
        markerstrokecolor=:black)
end

model_name = "k50_reg(1e-2)"
mean_explain_df = CSV.read("result/xai/camels_ig_values_$(model_name).csv", DataFrame)
total_xai_vec = mean_explain_df[!, :S1] .+ mean_explain_df[!, :Melt] .+ mean_explain_df[!, :Rainfall]
max_s1_basin_id = lpad(mean_explain_df[argmax(mean_explain_df.S1 ./ total_xai_vec), :basin_id], 8, "0")
max_melt_basin_id = lpad(mean_explain_df[argmax(mean_explain_df.Melt ./ total_xai_vec), :basin_id], 8, "0")
max_rainfall_basin_id = lpad(mean_explain_df[argmax(mean_explain_df.Rainfall ./ total_xai_vec), :basin_id], 8, "0")

fig1 = plot(size=(300, 300), fontfamily="Times", legend=false, framestyle=:box, grid=true, dpi=300)
fig2 = plot(size=(300, 300), fontfamily="Times", legend=false, framestyle=:box, grid=true, dpi=300)
fig3 = plot(size=(300, 300), fontfamily="Times", legend=false, framestyle=:box, grid=true, dpi=300)

plot_one_xai_scatter_m50!(fig1, max_melt_basin_id, "melt")
plot_one_xai_scatter_m50!(fig2, max_s1_basin_id, "s1")
plot_one_xai_scatter_m50!(fig3, max_rainfall_basin_id, "rainfall")

plot_one_xai_scatter_k50!(fig1, max_melt_basin_id, "melt", "k50_reg(1e-2)")
plot_one_xai_scatter_k50!(fig2, max_s1_basin_id, "s1", "k50_reg(1e-2)")
plot_one_xai_scatter_k50!(fig3, max_rainfall_basin_id, "rainfall", "k50_reg(1e-2)")

savefig(fig1, "result/xai/plots/k50_melt_$(max_melt_basin_id).png")
savefig(fig2, "result/xai/plots/k50_s1_$(max_s1_basin_id).png")
savefig(fig3, "result/xai/plots/k50_rainfall_$(max_rainfall_basin_id).png")


