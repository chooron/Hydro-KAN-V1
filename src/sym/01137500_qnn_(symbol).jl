using Symbolics
using ComponentArrays
using Plots
using Statistics
using Lux
using StableRNGs
using Random

include("../utils/symbolic_relate.jl")
include("../utils/kan_tools.jl")
include("../utils/data_relate.jl")

@variables x1
model_name = "k50_prune(1e-2)"
basin_id = "01137500"
formula_path = "result/formulas/$(model_name)/$(basin_id)/qnn"
layer_eqs = extract_layer_eqs(formula_path)
# 0.81
q_func_111, params_111 = parse_to_func(layer_eqs["layer1_I1_A1"][6, :Equation], params_nm=:p111)
q_func_121, params_121 = parse_to_func(layer_eqs["layer1_I2_A1"][4, :Equation], params_nm=:p121)
q_func_131, params_131 = parse_to_func(layer_eqs["layer1_I3_A1"][7, :Equation], params_nm=:p131)
# layer 2
q_func_211, params_211 = parse_to_func(layer_eqs["layer2_I1_A1"][6, :Equation], params_nm=:p211)
q_params = ComponentArray(reduce(merge, (params_111, params_121, params_131, params_211)))

function load_splines(basin_id)
    node_num = 1
    qnn = Chain(
        KDense(3, node_num, 6; use_base_act=true),
        KDense(node_num, 1, 6; use_base_act=true),
        name=:qnn,
    )
    qnn_ps_axes = getaxes(ComponentArray(Lux.initialparameters(Random.default_rng(), qnn)))
    output_df = load("result/$(model_name)/$basin_id/other_info.jld2")["output_df"]
    qnn_input = output_df[365:end, [:norm_snw, :norm_slw, :norm_infil]] |> Array |> permutedims
    flow_vec = output_df[365:end, [:flow]] |> Array |> vec

    train_params = load("result/$(model_name)/$(basin_id)/train_records.jld2")
    q_pas = ComponentArray(train_params["reg_opt_ps"][:nns][:qnn], qnn_ps_axes)
    @info reg_loss(1.0, [:layer_1, :layer_2])(q_pas)
    qnn_states = LuxCore.initialstates(StableRNG(1234), qnn)
    qnn_layer1_postacts = activation_getter(qnn.layer_1, q_pas.layer_1, qnn_states.layer_1, qnn_input)
    qnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), qnn_layer1_postacts)
    qnn_layer1_output = qnn.layer_1(qnn_input, q_pas.layer_1, qnn_states.layer_1)[1]
    qnn_layer2_postacts = activation_getter(qnn.layer_2, q_pas.layer_2, qnn_states.layer_2, qnn_layer1_output)
    qnn_layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), qnn_layer2_postacts)
    return (qnn_input, qnn_layer1_output), (qnn_layer1_postacts_arr, qnn_layer2_postacts_arr)
end

function check_fit_plot(tmp_func, params, input, target; title="", figsize=(400, 400))
    # Sort input and get sorting indices
    sorted_indices = sortperm(input)
    sorted_target = target[sorted_indices]
    sorted_input = input[sorted_indices]
    x_min, x_max = extrema(sorted_input)

    # Calculate predictions and R2
    predictions = tmp_func.(sorted_input, Ref(params))
    ss_res = sum((sorted_target .- predictions) .^ 2)
    ss_tot = sum((sorted_target .- mean(sorted_target)) .^ 2)
    r2 = 1 - ss_res / ss_tot

    fig = plot(size=figsize)
    # Plot predicted line
    plot!(
        sorted_input, predictions,
        fontsize=14, guidefontsize=16, tickfont=font(14, "Times"), dpi=300,
        linewidth=2, color=:black, label="Predicted", legendfontsize=14,
        bottom_margin=5Plots.mm
    )
    # Plot target points  
    scatter!(sorted_input, sorted_target, dpi=300, markerstrokewidth=0.2,
        color=:salmon, markersize=3, label="Target", alpha=0.2,
        xticks=round.(Int, range(floor(x_min), ceil(x_max), length=5))
    )
    # Add title with R2
    title!("$(title)", fontsize=16, fontfamily="Times")
    # Add R² annotation in bottom right corner
    bound_v = max(maximum(predictions), maximum(sorted_target)) - min(minimum(predictions), minimum(sorted_target))
    annote_pos = (maximum(sorted_input), max(maximum(predictions), maximum(sorted_target)) - 0.9 * bound_v)
    annotate!([(annote_pos..., Plots.text("R² = $(floor(r2, digits=2))", 14, "Times", :right, :bottom))])
    # Add x-axis and y-axis labels

    return fig
end

function plot_q_symbolize_fit(basin_id)
    qnn_acts, qnn_postacts = load_splines(basin_id)
    q_input, qnn_layer1_output = qnn_acts
    qnn_layer1_postacts_arr, qnn_layer2_postacts_arr = qnn_postacts

    # Layer 1 plots
    fig111 = check_fit_plot(q_func_111, params_111, q_input[1, :], qnn_layer1_postacts_arr[:, 1, 1], figsize=(300, 300))
    fig121 = check_fit_plot(q_func_121, params_121, q_input[2, :], qnn_layer1_postacts_arr[:, 1, 2], figsize=(300, 300))
    fig131 = check_fit_plot(q_func_131, params_131, q_input[3, :], qnn_layer1_postacts_arr[:, 1, 3], figsize=(300, 300))
    # Layer 2 plots  
    fig211 = check_fit_plot(q_func_211, params_211, qnn_layer1_output[1, :], qnn_layer2_postacts_arr[:, 1, 1], figsize=(300, 300))

    # Combine plots
    layer1_plot = plot(
        fig111, fig121, fig131, fig211, framestyle=:box,
        layout=(1, 4), size=(1200, 300), legend=false, fontcolor=:black,
        fontfamily="Times", tickfont=font(14, "Times"), dpi=300
    )

    return layer1_plot
end

function plot_q_symbolize_predict(basin_id)
    output_df = load("result/$(model_name)/$basin_id/other_info.jld2")["output_df"]
    qnn_input = output_df[365:end, [:norm_snw, :norm_slw, :norm_infil]] |> Array |> permutedims
    flow_vec = output_df[365:end, [:flow]] |> Array |> vec
    model = QNNSymbol1(:qnn)
    ps, st = Lux.setup(StableRNG(1234), model)
    model_output = model(qnn_input, ps, st)[1]
    return flow_vec, model_output[1, :]
end

(qnn_input, qnn_layer1_output), (qnn_layer1_postacts_arr, qnn_layer2_postacts_arr) = load_splines(basin_id)
layer2_input_extrema = extrema(sum(qnn_layer1_postacts_arr, dims=3)[:,1,1])

struct QNNSymbol1 <: LuxCore.AbstractLuxLayer
    name::Symbol
end

LuxCore.initialparameters(::AbstractRNG, ::QNNSymbol1) = ComponentArray(reduce(merge, (params_121, params_131, params_211)))
LuxCore.initialstates(::AbstractRNG, ::QNNSymbol1) = NamedTuple()

function (::QNNSymbol1)(x::AbstractVector, p, st)
    acts2_output = q_func_121(x[2], p)
    acts3_output = q_func_131(x[3], p)
    layer1_output = clamp(acts2_output + acts3_output, layer2_input_extrema...)
    return [q_func_211(layer1_output, p)], st
end

function (::QNNSymbol1)(x::AbstractMatrix, p, st)
    acts2_output = q_func_121.(x[2, :], Ref(p))
    acts3_output = q_func_131.(x[3, :], Ref(p))
    layer1_output = clamp.(acts2_output .+ acts3_output, layer2_input_extrema...)
    output = q_func_211.(layer1_output, Ref(p))
    return reshape(output, 1, :), st
end

p = plot_q_symbolize_fit(basin_id)
# savefig(p, "src/plots/figures/$(basin_id)_qnn_symbolize_fit.png")