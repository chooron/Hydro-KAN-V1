# 根据筛选条件：
# 1. lbfgs_reg_loss < 0.5 * init_reg_loss
# 2. reg_nse > 0.6
# 筛选出符合条件的样本
using CSV, DataFrames
using Lux, Random, ComponentArrays
include("../utils/train.jl")
include("../utils/kan_tools.jl")
include("../utils/data_relate.jl")
include("../models/m50.jl")

reg_gamma = "1e-2"
model_name = "k50_reg($reg_gamma)"
base_path = "src/stats/$(model_name)_loss_df.csv"
loss_df = CSV.read(base_path, DataFrame)
loss_filtered = loss_df[loss_df.lbfgs_reg_loss.<0.5 .* loss_df.init_reg_loss, :].basin_id
criteria_df = CSV.read("src/stats/$(model_name)-criteria.csv", DataFrame)
criteria_filtered = criteria_df[criteria_df[!, Symbol("nse-test")].>0.6, :].station_id
basin_filtered = lpad.(intersect(loss_filtered, criteria_filtered), 8, "0")

original_et_nn, _ = build_nns()
etnn_ps_axes = getaxes(LuxCore.initialparameters(Random.default_rng(), original_et_nn) |> ComponentArray)
basin_id = basin_filtered[1]
keep_node_nums = []
for basin_id in basin_filtered
    etnn_input,_ = load_nn_data(basin_id, "exphydro(516)")
    kan_ckpt = load("result/$(model_name)/$(basin_id)/train_records.jld2")["reg_opt_ps"]
    pruned_et_nn_layers, pruned_et_ps, nodes_to_keep, node_scores = prune_etnn_nodes(
        model_layers=[original_et_nn.layer_1, original_et_nn.layer_2],
        layer_params=ComponentVector(kan_ckpt["nns"]["etnn"], etnn_ps_axes),
        input_data=etnn_input, prune_threshold=1 / 3,
    )
    keep_node_num = length(nodes_to_keep)
    push!(keep_node_nums, keep_node_num)
end

nums_stats = map(1:3) do i
    count(keep_node_nums .== i)
end
bar(1:3, nums_stats, xlabel="Number of nodes", ylabel="Count", title="Histogram of number of nodes")


