using CSV, DataFrames, Statistics, JLD2
using Plots

include("../utils/data_relate.jl")

reg_gamma = "1e-2"
base_path = "result/k50_prune($(reg_gamma))"
available_basins = readdir(base_path)
node_nums = map(available_basins) do basin_id
    tmp_other_info = load(joinpath(base_path, basin_id, "other_info.jld2"))
    tmp_other_info["qnn_nodes_to_keep"] |> length
end
node_num_df = map(zip(available_basins, node_nums)) do (basin_id, node_num)
    (basin_id=basin_id, node_num=node_num)
end |> DataFrame
CSV.write("src/stats/k50_prune($(reg_gamma))-node_nums.csv", node_num_df)
count_list = map(1:6) do i
    filter(x -> x == i, node_nums) |> length
end

combine_fig1 = bar(count_list, label="", ylims=(0.5, 6.5),
    framestyle=:box, orientation=:horizontal, bar_width=0.5,
    color=[:skyblue, :salmon, :mediumseagreen, :mediumpurple, :grey, :grey],
    legend=false, size=(300, 500), margin=0.2Plots.mm,
)

function get_node_scores(basin_id, reg_gamma="1e-2")
    qnn_input = load_nn_data(basin_id)[2]
    model_name = "k50_reg($(reg_gamma))"
    #* build model
    original_et_nn, original_q_nn = build_nns()
    qnn_ps_axes = getaxes(LuxCore.initialparameters(Random.default_rng(), original_q_nn) |> ComponentArray)
    #* prune nodes
    kan_ckpt = load("result/$(model_name)/$(basin_id)/train_records.jld2")["reg_opt_ps"]
    pruned_q_nn_layers, pruned_q_ps, qnn_nodes_to_keep, qnn_node_scores = prune_qnn_nodes(
        model_layers=[original_q_nn.layer_1, original_q_nn.layer_2],
        layer_params=ComponentVector(kan_ckpt["nns"]["qnn"], qnn_ps_axes),
        input_data=qnn_input, prune_threshold=1 / 6,
    )
    return qnn_node_scores
end

qnn_node_scores1 = get_node_scores("01137500")
qnn_node_scores2 = get_node_scores("01134500")
qnn_node_scores3 = get_node_scores("01030500")
qnn_node_scores4 = get_node_scores("01532000")

palette1 = (:skyblue, :salmon, :mediumseagreen, :mediumpurple)
fig1 = bar(qnn_node_scores1, label="01137500", color=palette1[1], ymirror=true)
mean1 = mean(qnn_node_scores1)
hline!(fig1, [mean1], label="Mean", color=:red, linestyle=:dash, linewidth=3)
fig2 = bar(qnn_node_scores2, label="01134500", color=palette1[2], yticks=[0, 1, 2, 3], ylims=(0, 3), ymirror=true)
mean2 = mean(qnn_node_scores2)
hline!(fig2, [mean2], label="Mean", color=:red, linestyle=:dash, linewidth=3)
fig3 = bar(qnn_node_scores3, label="01030500", color=palette1[3], ymirror=true)
mean3 = mean(qnn_node_scores3)
hline!(fig3, [mean3], label="Mean", color=:red, linestyle=:dash, linewidth=3)
fig4 = bar(qnn_node_scores4, label="01532000", color=palette1[4], ymirror=true)
mean4 = mean(qnn_node_scores4)
hline!(fig4, [mean4], label="Mean", color=:red, linestyle=:dash, linewidth=3)

combine_fig2 = plot(
    fig1, fig2, fig3, fig4, layout=(2, 2), framestyle=:box,
)

fig = plot(
    combine_fig1, combine_fig2, layout=(1, 2), dpi=300,
    legend=false, size=(900, 400), margin=0.2Plots.mm,
    tickfontsize=14, legendfontsize=14, fontfamily="Times"
)

savefig(fig, "src/plots/figures/k50_prune($(reg_gamma))-node_scores.png")
