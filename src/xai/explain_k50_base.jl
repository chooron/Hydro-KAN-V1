# 使用ExplainableAI对模型的
using Lux
using Zygote
using JLD2, CSV, DataFrames
using ExplainableAI
using StableRNGs
using Statistics
using Plots

include("../models/m50.jl")
include("../utils/data_relate.jl")


# function main(basin_id, model_name)
basin_id = "02361000"
model_name = "k50_reg(1e-2)"
et_input, qnn_input = load_nn_data(basin_id, "exphydro(516)")
etnn, qnn = build_nns()
# qnn_train_params = load("result/v2/k50_base/$(basin_id)/train_records.jld2")["opt_ps"][:nns][:qnn]
qnn_train_params = load("result/$(model_name)/$(basin_id)/train_records.jld2")["opt_ps"][:nns][:qnn]
qnn_ps, qnn_st = Lux.setup(StableRNG(42), qnn)
qnn_ps_axes = getaxes(ComponentArray(qnn_ps))
qnn_func(x) = qnn(x, ComponentArray(qnn_train_params, qnn_ps_axes), qnn_st)[1]

# analyzer2 = IntegratedGradients(qnn_func)
analyzer2 = SmoothGrad(qnn_func)
# expl1 = analyze(qnn_input, analyzer1)
expl2 = analyze(qnn_input, analyzer2)

# exp1_df = DataFrame((s0=expl1.val[1, :], s1=expl1.val[2, :], infil=expl1.val[3, :], output=expl1.output[1, :]))
exp2_df = DataFrame((s0=expl2.val[1, :], s1=expl2.val[2, :], infil=expl2.val[3, :], output=expl2.output[1, :]))
# sort!(exp1_df, 4)
sort!(exp2_df, 4)

# scatter(exp1_df.output, exp1_df.s0, label="S0", alpha=0.5, markerstrokewidth=0)
# scatter!(exp1_df.output, exp1_df.s1, label="S1", alpha=0.5, markerstrokewidth=0)
# scatter!(exp1_df.output, exp1_df.infil, label="I", alpha=0.5, markerstrokewidth=0)

scatter(exp2_df.output, exp2_df.s0, label="S0", alpha=0.5, markerstrokewidth=0)
scatter!(exp2_df.output, exp2_df.s1, label="S1", alpha=0.5, markerstrokewidth=0)
scatter!(exp2_df.output, exp2_df.infil, label="I", alpha=0.5, markerstrokewidth=0)

