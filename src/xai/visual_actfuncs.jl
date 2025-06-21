# 可视化激活函数
using Lux
using JLD2, ComponentArrays
using SymbolicRegression
using Symbolics
using SymbolicUtils
using Random
include("../utils/kan_tools.jl")
include("../utils/symbolic_relate.jl")
include("../utils/train.jl")
include("../utils/data_relate.jl")
include("../models/m50.jl")

model_name = "k50_prune(1e-2)"
basin_id = "01137500" # "01022500"
et_nn, q_nn = build_nns()
etnn_ps_axes = getaxes(ComponentArray(Lux.initialparameters(Random.default_rng(), et_nn)))
qnn_ps_axes = getaxes(ComponentArray(Lux.initialparameters(Random.default_rng(), q_nn)))
et_input, qnn_input = load_nn_data(basin_id, "exphydro(516)")

train_params = load("result/$(model_name)/$(basin_id)/train_records.jld2")
loss_dfs = load("result/$(model_name)/$(basin_id)/loss_df.jld2")
# et_pas = ComponentArray(train_params["reg_opt_ps"][:nns][:etnn], etnn_ps_axes)
et_pas = ComponentArray(train_params["reg_opt_ps"], etnn_ps_axes)
et_nn_states = LuxCore.initialstates(StableRNG(1234), et_nn)
etnn_layer1_postacts = activation_getter(et_nn.layer_1, et_pas.layer_1, et_nn_states.layer_1, et_input)
etnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), etnn_layer1_postacts)
p1 = plot_activation_functions(et_input, etnn_layer1_postacts_arr)

q_pas = ComponentArray(train_params["reg_opt_ps"][:nns][:qnn], qnn_ps_axes)
@info reg_loss(1.0, [:layer_1, :layer_2])(q_pas)
q_nn_states = LuxCore.initialstates(StableRNG(1234), q_nn)
qnn_layer1_postacts = activation_getter(q_nn.layer_1, q_pas.layer_1, q_nn_states.layer_1, qnn_input)
qnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), qnn_layer1_postacts)
qnn_layer1_output = q_nn.layer_1(qnn_input, q_pas.layer_1, q_nn_states.layer_1)[1]
qnn_layer2_postacts = activation_getter(q_nn.layer_2, q_pas.layer_2, q_nn_states.layer_2, qnn_layer1_output)
qnn_layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), qnn_layer2_postacts)
p2 = plot_activation_functions(qnn_input, qnn_layer1_postacts_arr)
p3 = plot_activation_functions(qnn_layer1_output, qnn_layer2_postacts_arr)