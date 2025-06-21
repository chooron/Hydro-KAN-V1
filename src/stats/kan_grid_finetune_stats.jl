using JSON
using DataFrames, CSV, DelimitedFiles
using Plots

basin_file = readdlm(joinpath("data/basin_ids/basins_all.txt"), ',')
basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")

df_list = []
for grid_size in [4, 5, 6, 7, 8]
    json_list = []
    for basin_id in basins_available
        open("result/finetune/kan(q)_6_$(grid_size)/$(basin_id).json", "r") do f
            data = JSON.parse(f)
            push!(json_list, data)
        end
    end
    df = DataFrame(json_list)
    push!(df_list, df)
end

stats_list = []
for (grid_size, df) in zip([4, 5, 6, 7, 8], df_list)
    @info "mean train_mse: $(mean(df.train_mse)), mean test_mse: $(mean(df.test_mse))"
    push!(stats_list, Dict(
        "hidd_dims" => 6, "grid_size" => grid_size,
        "mean_train_mse" => mean(df.train_mse), "mean_test_mse" => mean(df.test_mse),
        "median_train_mse" => median(df.train_mse), "median_test_mse" => median(df.test_mse),
        "std_train_mse" => std(df.train_mse), "std_test_mse" => std(df.test_mse),
    ))
    # 创建组合图
    p = plot(layout=(1, 2), size=(1200, 500))

    # 训练MSE直方图
    histogram!(p[1], log.(df.train_mse),
        bins=20,
        label="Train-MSE",
        xlabel="MSE",
        ylabel="Frequency",
        title="Train-MSE Distribution",
        color=:blue,
        alpha=0.6)

    # 测试MSE直方图
    histogram!(p[2], log.(df.test_mse),
        bins=20,
        label="Test-MSE",
        xlabel="MSE",
        ylabel="Frequency",
        title="Test-MSE Distribution",
        color=:red,
        alpha=0.6)

    savefig(p, "result/finetune/plots/kan(q)_6_$(grid_size)_mse_dist.png")
end
stat_df = DataFrame(stats_list)
stat_df = stat_df[!, ["hidd_dims", "grid_size", "mean_train_mse", "mean_test_mse", "median_train_mse", "median_test_mse", "std_train_mse", "std_test_mse"]]
CSV.write("result/finetune/kan(q)_grid_stats.csv", stat_df)
