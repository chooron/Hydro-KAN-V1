# 根据筛选条件：
# 1. lbfgs_reg_loss < 0.5 * init_reg_loss
# 2. reg_nse > 0.6
# 筛选出符合条件的样本
using CSV, DataFrames
using Lux, Random, ComponentArrays
using Plots, StatsPlots, Statistics
include("../utils/train.jl")
include("../utils/kan_tools.jl")
include("../utils/data_relate.jl")
include("../models/m50.jl")

reg_gamma = "1e-2"
model_name = "k50_reg($reg_gamma)"
base_path = "src/v2/stats/$(model_name)_loss_df.csv"
loss_df = CSV.read(base_path, DataFrame)
loss_filtered = loss_df[loss_df.lbfgs_reg_loss.<0.5 .* loss_df.init_reg_loss, :].basin_id
criteria_df = CSV.read("src/v2/stats/$(model_name)-criteria.csv", DataFrame)
criteria_filtered = criteria_df[criteria_df[!, Symbol("nse-test")].>0.6, :].station_id
basin_filtered = lpad.(intersect(loss_filtered, criteria_filtered), 8, "0")

_, original_q_nn = build_nns()
qnn_ps_axes = getaxes(LuxCore.initialparameters(Random.default_rng(), original_q_nn) |> ComponentArray)
basin_id = basin_filtered[1]
keep_node_nums = []
node_scores_list = []
for basin_id in basin_filtered
    _, qnn_input = load_nn_data(basin_id, "exphydro(516)")
    kan_ckpt = load("result/v2/$(model_name)/$(basin_id)/train_records.jld2")["reg_opt_ps"]
    qnn_prune_threshold = 1 / 6
    pruned_q_nn_layers, pruned_q_ps, nodes_to_keep, node_scores = prune_qnn_nodes(
        model_layers=[original_q_nn.layer_1, original_q_nn.layer_2],
        layer_params=ComponentVector(kan_ckpt["nns"]["qnn"], qnn_ps_axes),
        input_data=qnn_input, prune_threshold=qnn_prune_threshold,
    )
    keep_node_num = length(nodes_to_keep)
    push!(keep_node_nums, keep_node_num)
    if keep_node_num <= 2
        push!(node_scores_list, node_scores)
    end
end
nums_stats = map(1:6) do i
    count(keep_node_nums .== i)
end
bar(1:6, nums_stats, xlabel="Number of nodes", ylabel="Count", title="Histogram of number of nodes")

# node_scores = stack(node_scores_list, dims=1)
# sum_node_scores = sum(node_scores, dims=2)

# heatmap((node_scores ./ sum_node_scores) |> permutedims, size=(1200, 400))
# # 绘制箱线图
# boxplot(1:6, eachcol(node_scores ./ sum_node_scores), 
#     xlabel="Node Index", 
#     ylabel="Normalized Score",
#     title="Distribution of Node Scores",
#     label="",
#     xticks=1:6)

