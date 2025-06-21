using JLD2
using CSV
using DataFrames
using Statistics
using Plots

model_name = "k50_reg"
basin_id = "01350140"
model_dir = "result/$(model_name)/$(basin_id)"

loss_df = load("$(model_dir)/loss_df.jld2")

plot(1:200, loss_df["adam_loss_df"][1:200, :train_loss], label="adam train loss", yscale=:log10)
plot!(1:200, loss_df["adam_loss_df"][1:200, :val_loss], label="adam val loss", yscale=:log10)
plot!(201:300, loss_df["lbfgs_loss_df"][1:100, :train_loss], label="lbfgs train loss", yscale=:log10)
plot!(201:300, loss_df["lbfgs_loss_df"][1:100, :val_loss], label="lbfgs val loss", yscale=:log10)