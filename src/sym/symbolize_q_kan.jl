# 拟合KAN的激活曲线然后构建符号公式


using Lux
using JLD2
using SymbolicRegression
using Symbolics
using SymbolicUtils
using KolmogorovArnold
using Random

include("../utils/kan_tools.jl")
include("../utils/symbolic_relate.jl")
include("../utils/data_relate.jl")

# basin_id = "03281500"  # 02361000, 03281500, 06191500  # call by other file
model_name = "k50_prune(1e-2)"
basin_id = "03281500"
node_num = 3
qnn = Chain(
    KDense(3, node_num, 6; use_base_act=true),
    KDense(node_num, 1, 6; use_base_act=true),
    name=:qnn,
)
qnn_ps_axes = getaxes(ComponentArray(Lux.initialparameters(Random.default_rng(), qnn)))
qnn_st = Lux.initialstates(Random.default_rng(), qnn)
output_df = load("result/v2/$(model_name)/$basin_id/other_info.jld2")["output_df"]
qnn_input = output_df[365:end, [:norm_snw, :norm_slw, :norm_infil]] |> Array |> permutedims
flow_vec = output_df[365:end, [:flow]] |> Array |> vec

train_params = load("result/v2/$(model_name)/$(basin_id)/train_records.jld2")
q_pas = ComponentArray(train_params["reg_opt_ps"][:nns][:qnn], qnn_ps_axes)
@info reg_loss(1.0, [:layer_1, :layer_2])(q_pas)
qnn_states = LuxCore.initialstates(StableRNG(1234), qnn)
qnn_layer1_postacts = activation_getter(qnn.layer_1, q_pas.layer_1, qnn_states.layer_1, qnn_input)
qnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), qnn_layer1_postacts)
qnn_layer1_output = qnn.layer_1(qnn_input, q_pas.layer_1, qnn_states.layer_1)[1]
qnn_layer2_postacts = activation_getter(qnn.layer_2, q_pas.layer_2, qnn_states.layer_2, qnn_layer1_output)
qnn_layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), qnn_layer2_postacts)
mkpath("src/v2/formulas/$(model_name)/$(basin_id)/qnn")
fit_kan_activations(qnn, qnn_input, q_pas, "src/v2/formulas/$(model_name)/$(basin_id)/qnn")