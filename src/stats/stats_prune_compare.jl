using CSV, DataFrames, Statistics, JLD2
using Plots
using Printf

function plot_node_num(node_num, criteria, alpha=1.0)
    node_num_df = CSV.read("src/stats/k50_prune(1e-2)-node_nums.csv", DataFrame)
    select_basin_id = lpad.(node_num_df[node_num_df[!, :node_num].==node_num, :basin_id], 8, "0")
    prune_criteria = CSV.read("src/stats/k50_prune(1e-2)-criteria.csv", DataFrame)
    reg_criteria = CSV.read("src/stats/k50_reg(1e-2)-criteria.csv", DataFrame)
    prune_criteria[!, :station_id] = lpad.(string.(Int.(prune_criteria[!, :station_id])), 8, "0")
    prune_criteria = filter(x -> x.station_id in select_basin_id, prune_criteria)
    reg_criteria[!, :station_id] = lpad.(string.(Int.(reg_criteria[!, :station_id])), 8, "0")
    reg_criteria = filter(x -> x.station_id in select_basin_id, reg_criteria)

    prune_test_nse = prune_criteria[!, Symbol("$(criteria)-test")] |> Array |> sort
    reg_test_nse = reg_criteria[!, Symbol("$(criteria)-test")] |> Array |> sort
    bounds = (min(minimum(reg_test_nse), minimum(prune_test_nse))-0.01, (max(maximum(reg_test_nse), maximum(prune_test_nse)))+0.01)
    fig1 = plot(
        reg_test_nse,1:length(prune_test_nse), color=:salmon,
        linewidth=3, tickfontsize=14, legend=false,
        xlims=bounds, xticks = (floor(bounds[1], digits=1):0.1:ceil(bounds[2], digits=1)), xformatter = x -> @sprintf("%.1f", x),
    )
    plot!(prune_test_nse, 1:length(prune_test_nse), color=:skyblue, linewidth=3)
    plot!( reg_test_nse, 1:length(reg_test_nse),seriestype=:scatter, label="", color=:salmon, alpha=alpha, markerstrokewidth=1)
    plot!(prune_test_nse, 1:length(prune_test_nse), seriestype=:scatter, label="", color=:skyblue, alpha=alpha, markerstrokewidth=1)
    return fig1
end

figure_1 = plot(
    plot_node_num(1, "nse"),
    plot_node_num(2, "nse", 0.6),
    plot_node_num(3, "nse", 0.6),
    plot_node_num(4, "nse"),
    layout=(1, 4), size=(800, 200),
    bottom_margin=5Plots.mm, margin=0.2Plots.mm, fontfamily="Times", framestyle=:box
)

figure_2 = plot(
    plot_node_num(1, "mnse"),
    plot_node_num(2, "mnse", 0.6),
    plot_node_num(3, "mnse", 0.6),
    plot_node_num(4, "mnse"),
    layout=(1, 4), size=(800, 200),
    bottom_margin=5Plots.mm, margin=0.2Plots.mm, fontfamily="Times", framestyle=:box
)

fig = plot(figure_1, figure_2, layout=(2, 1), size=(800, 400), dpi=300,
    bottom_margin=5Plots.mm, margin=0.2Plots.mm, fontfamily="Times", framestyle=:box
)

savefig(fig, "src/plots/figures/k50_prune_perform_compare.png")
