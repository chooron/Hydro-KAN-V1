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

function main(basin_id, model_name)
    @info "Loading data for basin $basin_id"
    etnn, qnn = build_nns()
    qnn_train_params = load("result/$(model_name)/$basin_id/train_records.jld2")["reg_opt_ps"][:nns][:qnn]
    output_df = load("result/$(model_name)/$basin_id/other_info.jld2")["output_df"]
    qnn_input = output_df[365:end, [:norm_snw, :norm_slw, :norm_infil]] |> Array |> permutedims

    flow_vec = output_df[365:end, [:flow]] |> Array |> vec
    indices, _ = findmaxima(flow_vec)
    qnn_ps, qnn_st = Lux.setup(StableRNG(42), qnn)
    qnn_ps_axes = getaxes(ComponentArray(qnn_ps))
    qnn_func(x) = qnn(x, ComponentArray(qnn_train_params, qnn_ps_axes), qnn_st)[1]

    analyzer1, analyzer2 = IntegratedGradients(qnn_func), SmoothGrad(qnn_func)
    expl1 = analyze(qnn_input[:, indices], analyzer1)
    expl2 = analyze(qnn_input[:, indices], analyzer2)

    melt_vec = (output_df[365:end, [:melt]]|>Array|>vec)[indices]
    rainfall_vec = (output_df[365:end, [:rainfall]]|>Array|>vec)[indices]
    melt_prop_vec = melt_vec ./ (melt_vec .+ rainfall_vec)
    rainfall_prop_vec = rainfall_vec ./ (melt_vec .+ rainfall_vec)
    melt_prop_vec[isnan.(melt_prop_vec)] .= 0.0
    rainfall_prop_vec[isnan.(rainfall_prop_vec)] .= 0.0

    melt_contrib1 = melt_prop_vec .* expl1.val[3, :] # .+ expl1.val[1, :]
    rainfall_contrib1 = rainfall_prop_vec .* expl1.val[3, :]
    melt_contrib2 = melt_prop_vec .* expl2.val[3, :] # .+ expl2.val[1, :]
    rainfall_contrib2 = rainfall_prop_vec .* expl2.val[3, :]

    exp1_df = DataFrame((s0=expl1.val[1, :], s1=expl1.val[2, :], melt_contrib=melt_contrib1, rainfall_contrib=rainfall_contrib1, output=expl1.output[1, :]))
    exp2_df = DataFrame((s0=expl2.val[1, :], s1=expl2.val[2, :], melt_contrib=melt_contrib2, rainfall_contrib=rainfall_contrib2, output=expl2.output[1, :]))

    expl1_median = (basin_id=basin_id, S0=mean(abs.(exp1_df.s0)), S1=mean(abs.(exp1_df.s1)), Melt=mean(abs.(exp1_df.melt_contrib)), Rainfall=mean(abs.(exp1_df.rainfall_contrib)))
    expl2_median = (basin_id=basin_id, S0=mean(abs.(exp2_df.s0)), S1=mean(abs.(exp2_df.s1)), Melt=mean(abs.(exp2_df.melt_contrib)), Rainfall=mean(abs.(exp2_df.rainfall_contrib)))
    return expl1_median, expl2_median
end

basin_file = readdlm(joinpath("data/basin_ids/basins_all.txt"), ',')
basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")

ig_values_list = []
sg_values_list = []
model_name = "k50_reg(1e-2)"

for basin_id in basins_available
    ig_values, sg_values = main(basin_id, model_name)
    push!(ig_values_list, ig_values)
    push!(sg_values_list, sg_values)
end

camels_ig_values = DataFrame(ig_values_list)
camels_sg_values = DataFrame(sg_values_list)
CSV.write("src/xai/result/camels_ig_values_$(model_name).csv", camels_ig_values)
CSV.write("src/xai/result/camels_sg_values_$(model_name).csv", camels_sg_values)

