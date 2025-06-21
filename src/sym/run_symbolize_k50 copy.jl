using Lux
using Plots
using JLD2
using DifferentialEquations
using SciMLSensitivity
using StableRNGs
using ForwardDiff
using Statistics
using ComponentArrays
using Interpolations
using ProgressMeter
using ParameterSchedulers: Scheduler, Exp, Step
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using CSV, DataFrames, Dates, DelimitedFiles

include("../models/m50.jl")
include("../utils/train.jl")
basin_id = "01137500" # 02053200
include("$(basin_id)_qnn_(symbol).jl")
reg_gamma = "1e-2"
reg_gamma_dict = Dict("1e-2" => 1e-2, "1e-3" => 1e-3, "5e-3" => 5e-3)
model_dict = Dict("02053200" => QNNSymbol2(:qnn), "03281500" => QNNSymbol3(:qnn), "01137500" => QNNSymbol1(:qnn))
model_name = "k50_reg($reg_gamma)"
save_model_name = "k50_symbol($reg_gamma)"
basemodel_name = "exphydro(516)"
basemodel_dir = "result/$basemodel_name"
loss_df = load("result/k50_prune(1e-2)/$(basin_id)/loss_df.jld2")
loss_df["lbfgs_loss_df"]
#* load data
camelsus_cache = load("data/camelsus/$(basin_id).jld2")
data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
test_x, test_y, test_timepoints = camelsus_cache["test_x"], camelsus_cache["test_y"], camelsus_cache["test_timepoints"]
lday_vec, prcp_vec, temp_vec = collect(data_x[:, 1]), collect(data_x[:, 2]), collect(data_x[:, 3])
# Replace missing values with 0.0 in data arrays
for data_arr in [data_x, data_y, train_x, train_y, test_x, test_y]
    replace!(data_arr, missing => 0.0)
end

#* load parameters and initial states
exphydro_pas = load("$(basemodel_dir)/$(basin_id)/opt_params.jld2")["opt_params"]
exphydro_params = NamedTuple{(:f, :Smax, :Qmax, :Df, :Tmax, :Tmin)}(exphydro_pas[1:6]) |> ComponentArray
exphydro_init_states = NamedTuple{(:snowpack, :soilwater)}(exphydro_pas[7:8]) |> ComponentArray

#* load exphydro model outputs
exphydro_df = CSV.read("$(basemodel_dir)/$(basin_id)/model_outputs.csv", DataFrame)
snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
et_vec, flow_vec = exphydro_df[!, "et"], exphydro_df[!, "qsim"]
pr_vec, melt_vec = exphydro_df[!, "pr"], exphydro_df[!, "melt"]
infil_vec = pr_vec .+ melt_vec
et_vec[et_vec.<0] .= 0.000000001
flow_vec[flow_vec.<0] .= 0.000000001
infil_vec[infil_vec.<0] .= 0.000000001

#* normalize data
s0_mean, s0_std = mean(snowpack_vec), std(snowpack_vec)
s1_mean, s1_std = mean(soilwater_vec), std(soilwater_vec)
infil_mean, infil_std = mean(infil_vec), std(infil_vec)

stand_params = [s0_std, s0_mean, s1_std, s1_mean, infil_std, infil_mean]
# symbol_model = QNNSymbol1(:qnn)
# symbol_model = QNNSymbol2(:qnn)
symbol_model = model_dict[basin_id]
q_pas, _ = Lux.setup(Random.default_rng(), symbol_model)
m25_model = build_m25_model(symbol_model, stand_params)
model_init_pas = ComponentArray(
    nns=(qnn=Vector(ComponentVector(q_pas)),),
    params=exphydro_params
)
ps_axes = getaxes(model_init_pas)
# lday, prcp, temp => prcp, temp, lday
train_arr = permutedims(train_x)[[2, 3, 1], :]
test_arr = permutedims(test_x)[[2, 3, 1], :]
total_arr = permutedims(data_x)[[2, 3, 1], :]

initstates = ComponentArray(NamedTuple{(:snowpack, :soilwater)}(exphydro_init_states))
config = (
    solver=HydroModels.DiscreteSolver(),
    interp=DataInterpolations.LinearInterpolation
)
#* optimization
model_func(x, p) = m25_model(x, ComponentVector(p, ps_axes), initstates=initstates, config=config)[end, :]
result = m25_model(train_arr, model_init_pas, initstates=initstates, config=config)
train_best_ps, val_best_ps, train_recorder_df = train_hybrid(
    model_func,
    (train_arr, train_y, train_timepoints),
    (total_arr, data_y, data_timepoints),
    model_init_pas,
    loss_func=nse_loss,
    optmzr=LBFGS(linesearch=BackTracking()), max_N_iter=200, warm_up=365,
    adtype=Optimization.AutoForwardDiff()
)
output = m25_model(total_arr, train_best_ps, initstates=initstates, config=config)
output_names = Tuple(vcat(HydroModels.get_state_names(m25_model), HydroModels.get_output_names(m25_model)))
output_ntp = NamedTuple{output_names}(eachslice(output, dims=1))
output_ntp_merge = merge(output_ntp, (qreal=data_y,))
CSV.write("src/sym/output/$(basin_id)_output.csv", DataFrame(output_ntp_merge))
