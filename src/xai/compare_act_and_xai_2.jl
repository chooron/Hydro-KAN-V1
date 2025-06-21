# 比较激活函数和可解释性的贡献

using Lux
using JLD2, ComponentArrays
using SymbolicRegression
using Symbolics
using SymbolicUtils
using Random
using ExplainableAI
using Peaks
using Plots
using Zygote
include("../utils/kan_tools.jl")
include("../utils/symbolic_relate.jl")
include("../utils/train.jl")
include("../utils/data_relate.jl")

function plot_activation_functions(acts, postacts, colors=[:skyblue, :salmon, :mediumseagreen])
    input_dims = size(acts, 1)
    activ_dims = size(postacts, 2)
    p = plot(layout=(activ_dims, input_dims),
        size=(200 * input_dims, 200 * activ_dims),
        legend=false)
    for i in axes(acts, 1)
        current_acts = acts[i, :]
        sorted_indices = sortperm(current_acts)
        # sorted_indices = sorted_indices[ceil(Int, 0.1*length(sorted_indices)):ceil(Int, 0.9*length(sorted_indices))]
        for j in axes(postacts, 2)
            current_postacts = postacts[:, j, i]
            plot_idx = (j - 1) * input_dims + i
            plot!(
                p[plot_idx],
                current_acts[sorted_indices], current_postacts[sorted_indices],
                linewidth=4, color=colors[i], left_margin=i == 1 ? 0Plots.mm : 1Plots.mm # 5 for 01137500
            )
            scatter!(
                p[plot_idx],
                current_acts[sorted_indices[length(sorted_indices)-5:end]], current_postacts[sorted_indices[length(sorted_indices)-5:end]],
                markerstrokewidth=1.0, markerstrokecolor=:black, color=colors[i], left_margin=i == 1 ? 0Plots.mm : 5Plots.mm # 5 for 01137500
            )
            scatter!(
                p[plot_idx],
                current_acts[sorted_indices[1:5]], current_postacts[sorted_indices[1:5]],
                markerstrokewidth=1.0, markerstrokecolor=:black, color=colors[i], left_margin=i == 1 ? 0Plots.mm : 5Plots.mm # 5 for 01137500
            )
        end
    end
    plot!(p, fontfamily="Times", framestyle=:box, grid=true, bottom_margin=5Plots.mm, margin=0.1Plots.mm)
    return p
end


function plot_one_xai_scatter_m50!(fig, basin_id, var_type, color=:red)
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
    # melt_contrib1 = melt_prop_vec .* expl1.val[3, :] .+ expl1.val[1, :]
    melt_contrib1 = expl1.val[1, :]
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
    return scatter!(fig, x, y, alpha=0.5, color=color,
        tickfontsize=14, leftmargin=4Plots.mm, markerstrokewidth=1.0,
        markerstrokecolor=:black)
end

model_name = "k50_prune(1e-2)"
basin_id = "02053200"
node_num = 2
qnn = Chain(
    KDense(3, node_num, 6; use_base_act=true),
    KDense(node_num, 1, 6; use_base_act=true),
    name=:qnn,
)
qnn_ps_axes = getaxes(ComponentArray(Lux.initialparameters(Random.default_rng(), qnn)))
qnn_st = Lux.initialstates(Random.default_rng(), qnn)
output_df = load("result/$(model_name)/$basin_id/other_info.jld2")["output_df"]
qnn_input = output_df[365:end, [:norm_snw, :norm_slw, :norm_infil]] |> Array |> permutedims
max_args_infil = argmax(qnn_input[3, :])

tmp_output_df = output_df[365:end, :]
plot(tmp_output_df[max_args_infil-20:max_args_infil+20, :flow])
plot!(tmp_output_df[max_args_infil-20:max_args_infil+1, :flow])
bar!(tmp_output_df[max_args_infil-20:max_args_infil+1, :norm_infil])


flow_vec = output_df[365:end, [:flow]] |> Array |> vec
indices, _ = findmaxima(flow_vec)

train_params = load("result/$(model_name)/$(basin_id)/train_records.jld2")
q_pas = ComponentArray(train_params["reg_opt_ps"][:nns][:qnn], qnn_ps_axes)
@info reg_loss(1.0, [:layer_1, :layer_2])(q_pas)
qnn_states = LuxCore.initialstates(StableRNG(1234), qnn)
qnn_layer1_postacts = activation_getter(qnn.layer_1, q_pas.layer_1, qnn_states.layer_1, qnn_input)
qnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), qnn_layer1_postacts)
qnn_layer1_output = qnn.layer_1(qnn_input, q_pas.layer_1, qnn_states.layer_1)[1]
qnn_layer2_postacts = activation_getter(qnn.layer_2, q_pas.layer_2, qnn_states.layer_2, qnn_layer1_output)
qnn_layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), qnn_layer2_postacts)
p1 = plot_activation_functions(qnn_input, qnn_layer1_postacts_arr)
p2 = plot_activation_functions(qnn_layer1_output, qnn_layer2_postacts_arr, [:mediumpurple, :mediumpurple])

tmp_p2_1 = begin
    plot(p2.series_list[1], bottom_margin=5Plots.mm)
    plot!(p2.series_list[2], bottom_margin=5Plots.mm)
    plot!(p2.series_list[3], bottom_margin=5Plots.mm)
end
tmp_p2_2 = begin
    plot(p2.series_list[4], bottom_margin=5Plots.mm)
    plot!(p2.series_list[5], bottom_margin=5Plots.mm)
    plot!(p2.series_list[6], bottom_margin=5Plots.mm)
end
p2_t = plot(tmp_p2_1, tmp_p2_2, layout=(2, 1), size=(300, 600), dpi=300)

p_layout = @layout [grid(1, 2, widths=[3 / 4, 1 / 4])]
p = plot(p1, p2_t, layout=p_layout, fontfamily="Times", legend=false, framestyle=:box, grid=true, dpi=300, tickfontsize=14)

qnn_func(x) = qnn(x, q_pas, qnn_st)[1]
analyzer = IntegratedGradients(qnn_func)
expl = analyze(qnn_input[:, indices], analyzer)
scatter_colors = [:skyblue, :salmon, :mediumseagreen]
scatter_list = map(1:3) do i
    scatter(qnn_input[i, indices], expl.val[i, :], alpha=0.5, color=scatter_colors[i],
        tickfontsize=14, leftmargin=4Plots.mm, markerstrokewidth=1.0,
        markerstrokecolor=:black)
end
scatter_plot = plot(
    scatter_list..., layout=(1, 3), margin=0.1Plots.mm,
    fontfamily="Times", legend=false, framestyle=:box, grid=true, dpi=300
)

scatter_colors = [:skyblue, :salmon, :mediumseagreen]
fig1 = plot(size=(300, 300), fontfamily="Times", legend=false, framestyle=:box, grid=true, dpi=300)
plot_one_xai_scatter_m50!(fig1, basin_id, "melt", scatter_colors[1])
fig2 = plot(size=(300, 300), fontfamily="Times", legend=false, framestyle=:box, grid=true, dpi=300)
plot_one_xai_scatter_m50!(fig2, basin_id, "s1", scatter_colors[2])
fig3 = plot(size=(300, 300), fontfamily="Times", legend=false, framestyle=:box, grid=true, dpi=300)
plot_one_xai_scatter_m50!(fig3, basin_id, "rainfall", scatter_colors[3])
m50_xai_plot = plot(fig1, fig2, fig3, layout=(1, 3), size=(900, 300), dpi=300, topmargin=5Plots.mm)

#! m50_xai_plot is in plot_special_xai.jl
layout = @layout [grid(2, 1, heights=[3 / 5, 2 / 5])]
fig = plot(
    p, m50_xai_plot, layout=layout, size=(1200, 900), tickfontsize=14, margin=0.1Plots.mm,
    fontfamily="Times", legend=false, framestyle=:box, grid=true, dpi=600
)

savefig(fig, "src/v2/plots/figures/$(basin_id)_act_xai_compare.png")
