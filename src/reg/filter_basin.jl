# 根据筛选条件：
# 1. lbfgs_reg_loss < 0.5 * init_reg_loss
# 2. reg_nse > 0.6
# 筛选出符合条件的样本
using CSV, DataFrames

reg_gamma = "1e-2"
model_name = "k50_reg($reg_gamma)"
base_path = "src/stats/$(model_name)_loss_df.csv"
loss_df = CSV.read(base_path, DataFrame)
loss_filtered = loss_df[loss_df.lbfgs_reg_loss .< 0.5 .* loss_df.init_reg_loss, :].basin_id
criteria_df = CSV.read("src/stats/$(model_name)-criteria.csv", DataFrame)
criteria_filtered = criteria_df[criteria_df[!, Symbol("nse-test")] .> 0.6, :].station_id
basin_filtered = intersect(loss_filtered, criteria_filtered)

JLD2.save("src/cache/$(model_name)_basin_filtered.jld2", "basin_filtered", basin_filtered)
