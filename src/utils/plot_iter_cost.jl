# 绘制模型在训练过程中的损失曲线.
using CSV,DataFrames,Plots,Dates,StatsPlots

base_dir = "src/result/"

k50_avg_time_list = []
m50_avg_time_list = []
basin_file = readdlm(joinpath("src/data/basin_ids/basins_all.txt"), ',')
basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")

for basin_id in basins_available
    # Read loss data for M50 and K50 models
    m50_loss_df = CSV.read(joinpath(base_dir, "m50_f_d", basin_id, "loss_df.csv"), DataFrame)
    k50_loss_df = CSV.read(joinpath(base_dir, "k50_f_d", basin_id, "loss_df.csv"), DataFrame)
    # Calculate average computation time per iteration
    m50_times = DateTime.(m50_loss_df.time)
    k50_times = DateTime.(k50_loss_df.time)

    m50_time_diffs = [Dates.value(m50_times[i] - m50_times[i-1])/1000.0 for i in 2:length(m50_times)]
    k50_time_diffs = [Dates.value(k50_times[i] - k50_times[i-1])/1000.0 for i in 2:length(k50_times)]

    m50_avg_time = mean(m50_time_diffs)
    k50_avg_time = mean(k50_time_diffs)

    push!(k50_avg_time_list, k50_avg_time)
    push!(m50_avg_time_list, m50_avg_time)
end

function plot_time_comparison(k50_avg_time_list, m50_avg_time_list)
    # Create boxplot
    p = boxplot(
        ["K50" "M50"], 
        [k50_avg_time_list m50_avg_time_list],
        label=["K50" "M50"],
        ylabel="Time (seconds)",
        fillalpha=0.75,
        whisker_width=0.5,
        size=(300,450),
        outliers=true,
        marker=(:circle, 4),
        framestyle=:box,
        grid=false,
        mirror=true,
        left_margin=30Plots.px
    )
    p
end

# Customize plot
p = plot_time_comparison(k50_avg_time_list, m50_avg_time_list)

# Save plot
savefig(p, "src/plots/computation_time.png")




