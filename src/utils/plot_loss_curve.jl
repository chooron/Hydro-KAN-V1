# 绘制模型在训练过程中的损失曲线.
using CSV,DataFrames,Plots,StatsPlots

basin_id = "01030500"
base_dir = "src/result/"

# Read loss data for M50 and K50 models
m50_loss_df = CSV.read(joinpath(base_dir, "m50_f_d", basin_id, "loss_df.csv"), DataFrame)
k50_loss_df = CSV.read(joinpath(base_dir, "k50_f_d", basin_id, "loss_df.csv"), DataFrame)

# Create loss curve plot
p = plot(
    m50_loss_df.iter, m50_loss_df.train_loss, 
    label="M50-train",
    linewidth=2,
    alpha=0.8,
    dpi=300,
    size=(600, 400)
)

plot!(
    m50_loss_df.iter, m50_loss_df.val_loss,
    label="M50-test",
    linewidth=2,
    alpha=0.8
)

plot!(
    k50_loss_df.iter, k50_loss_df.train_loss,
    label="K50-train", 
    linewidth=2,
    alpha=0.8
)

plot!(
    k50_loss_df.iter, k50_loss_df.val_loss,
    label="K50-test",
    linewidth=2, 
    alpha=0.8
)

xlabel!("Iteration", fontsize=12)
ylabel!("Loss", fontsize=12)
plot!(framestyle=:box, grid=false)
plot!(legend=:topright)

# Save the plot
savefig(p, "src/plots/loss_curves/$(basin_id).png")

