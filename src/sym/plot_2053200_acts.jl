using Symbolics
using ComponentArrays
using Plots
using Statistics
using Lux
using StableRNGs

include("../utils/symbolic_relate.jl")
include("../utils/kan_tools.jl")
include("../utils/data_relate.jl")

@variables x1
model_name = "k50_prune(1e-2)"
basin_id = "02053200"
formula_path = "result/formulas/$(model_name)/$(basin_id)/qnn"
layer_eqs = extract_layer_eqs(formula_path)
# 0.81
q_func_111, params_111 = parse_to_func(layer_eqs["layer1_I1_A1"][6, :Equation], params_nm=:p111)
q_func_121, params_121 = parse_to_func(layer_eqs["layer1_I2_A1"][4, :Equation], params_nm=:p121)
q_func_131, params_131 = parse_to_func(layer_eqs["layer1_I3_A1"][6, :Equation], params_nm=:p131)
q_func_112, params_112 = parse_to_func(layer_eqs["layer1_I1_A2"][5, :Equation], params_nm=:p112)
q_func_122, params_122 = parse_to_func(layer_eqs["layer1_I2_A2"][4, :Equation], params_nm=:p122)
q_func_132, params_132 = parse_to_func(layer_eqs["layer1_I3_A2"][3, :Equation], params_nm=:p132)
# layer 2
q_func_211, params_211 = parse_to_func(layer_eqs["layer2_I1_A1"][12, :Equation], params_nm=:p211)
q_func_221, params_221 = parse_to_func(layer_eqs["layer2_I2_A1"][7, :Equation], params_nm=:p221)

output_df = load("result/v2/$(model_name)/$basin_id/other_info.jld2")["output_df"]
qnn_input = output_df[365:6000, [:norm_snw, :norm_slw, :norm_infil]] |> Array |> permutedims
flow_vec = output_df[365:6000, [:flow]] |> Array |> vec

common_sets = (
    xticks=:none, yticks=:none, legend=false,
    framestyle=:box, fontfamily="Times", tickfont=font(14, "Times"),
    dpi=300, linewidth=2, size=(300, 300), tickfontsize=16, margin=0.1Plots.mm
)

palette1 = (:skyblue, :salmon, :mediumseagreen)

p1 = plot(qnn_input[1, :], color=:salmon; common_sets...)
p2 = plot(qnn_input[2, :], color=:mediumseagreen; common_sets...)
p3 = plot(qnn_input[3, :], color=:skyblue; common_sets...)
savefig(p1, "src/v2/plots/figures/sym/$(basin_id)_snw.png")
savefig(p2, "src/v2/plots/figures/sym/$(basin_id)_slw.png")
savefig(p3, "src/v2/plots/figures/sym/$(basin_id)_infil.png")


q_func_121_output = q_func_121.(qnn_input[2, :], Ref(params_121))
q_func_131_output = q_func_131.(qnn_input[3, :], Ref(params_131))
q_func_11_output = q_func_121_output .+ q_func_131_output

fig121 = plot(q_func_121_output, label="", color=palette1[3]; common_sets...)
fig131 = plot(q_func_131_output, label="", color=palette1[1]; common_sets...)

q_func_122_output = q_func_122.(qnn_input[2, :], Ref(params_122))
q_func_132_output = q_func_132.(qnn_input[3, :], Ref(params_132))

fig122 = plot(q_func_122_output, label="", color=palette1[3]; common_sets...)
fig132 = plot(q_func_132_output, label="", color=palette1[1]; common_sets...)

q_func_211_output = q_func_211.(q_func_11_output, Ref(params_211))
q_func_221_output = q_func_221.(q_func_12_output, Ref(params_221))
q_func_21_output = q_func_211_output .+ q_func_221_output

fig211 = plot(q_func_211_output, label="", color=:darkgrey; common_sets...)
fig221 = plot(q_func_221_output, label="", color=:darkgrey; common_sets...)
fig21 = plot(q_func_21_output, label="", color=:black; common_sets...)

file_names = ["fig121", "fig131", "fig122", "fig132", "fig211", "fig221", "fig21"]
for (fig, file_name) in zip([fig121, fig131, fig122, fig132, fig211, fig221, fig21], file_names)
    savefig(fig, "src/v2/plots/figures/sym/$(basin_id)_$(file_name).png")
end

common_sets2 = (
    xticks=:none, yticks=:none, legend=false, xaxis=:none, yaxis=:none, framestyle=:none,
    fontfamily="Times", tickfont=font(14, "Times"),
    dpi=300, linewidth=16, size=(300, 300), tickfontsize=16, margin=0.1Plots.mm
)
for (func, func_name, param, idx) in zip(
    [q_func_121, q_func_131, q_func_122, q_func_132],
    ["q_func_121", "q_func_131", "q_func_122", "q_func_132"],
    [params_121, params_131, params_122, params_132],
    [2, 3, 2, 3]
)
    actfunc_output = func.(range(extrema(qnn_input[idx, :])..., length=1000), Ref(param))
    p = plot(actfunc_output, label="", color=:black; common_sets2...)
    savefig(p, "src/v2/plots/figures/sym/$(basin_id)_$(func_name).png")
end

for (func, func_name, param, output) in zip(
    [q_func_211, q_func_221],
    ["q_func_211", "q_func_221"],
    [params_211, params_221],
    [q_func_211_output, q_func_221_output]
)
    actfunc_output = func.(range(extrema(output)..., length=1000), Ref(param))
    p = plot(actfunc_output, label="", color=:black; common_sets2...)
    savefig(p, "src/v2/plots/figures/sym/$(basin_id)_$(func_name).png")
end