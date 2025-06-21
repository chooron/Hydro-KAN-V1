using JSON
using DataFrames, CSV, DelimitedFiles
using Plots
using Statistics, Measures
using Printf

# ADDED: Helper function to filter out non-finite values
function filter_finite(data)
    return filter(isfinite, data)
end

model_name = "kan(q)"
basin_file = readdlm(joinpath("data/basin_ids/basins_all.txt"), ',')
basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")

df_list = []
for reg_gamma in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    json_list = []
    for basin_id in basins_available
        open("result/finetune/$(model_name)_$(reg_gamma)/$(basin_id).json", "r") do f
            data = JSON.parse(f)
            push!(json_list, data)
        end
    end
    df = DataFrame(json_list)
    push!(df_list, df)
end

# ADDED: Calculate global x-axis limits
all_train_mse_logs = filter_finite(vcat([log.(df.train_mse) for df in df_list]...))
all_test_mse_logs = filter_finite(vcat([log.(df.test_mse) for df in df_list]...))
all_reg_loss_ratios = filter_finite(vcat([log.(df.final_reg_loss ./ df.init_reg_loss) for df in df_list]...))

# Helper function to calculate padded limits
function get_padded_limits(data_vector)
    if isempty(data_vector)
        return (0, 1) # Default if no finite data
    end
    min_val = minimum(data_vector)
    max_val = maximum(data_vector)
    if min_val == max_val
        return (min_val - 0.5, max_val + 0.5) # Handle single point case
    end
    padding = (max_val - min_val) * 0.05 # 5% padding
    return (min_val - padding, max_val + padding)
end

xlims_train_mse = get_padded_limits(all_train_mse_logs)
xlims_test_mse = get_padded_limits(all_test_mse_logs)
xlims_reg_loss_ratio = get_padded_limits(all_reg_loss_ratios)

stats_list = []
collected_plots_for_main_figure = []
reg_gamma_loop_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
num_rows = length(reg_gamma_loop_values)

for (idx, (reg_gamma, df)) in enumerate(zip(reg_gamma_loop_values, df_list))
    @info "mean train_mse: $(mean(df.train_mse)), mean test_mse: $(mean(df.test_mse))"
    push!(stats_list, Dict(
        "reg_gamma" => reg_gamma,
        "mean_train_mse" => mean(df.train_mse), "mean_test_mse" => mean(df.test_mse),
        "median_train_mse" => median(df.train_mse), "median_test_mse" => median(df.test_mse),
        "std_train_mse" => std(df.train_mse), "std_test_mse" => std(df.test_mse),
        "mean_init_reg_loss" => mean(df.init_reg_loss), "mean_final_reg_loss" => mean(df.final_reg_loss),
        "median_init_reg_loss" => median(df.init_reg_loss), "median_final_reg_loss" => median(df.final_reg_loss),
        "std_init_reg_loss" => std(df.init_reg_loss), "std_final_reg_loss" => std(df.final_reg_loss),
    ))
    # 创建组合图
    p = plot(
        layout=(1, 3), size=(1600, 500),
        guidefontsize=14, tickfontsize=12, legendfontsize=12, titlefontsize=14,
        title="γ = $(Printf.@sprintf("%.0e", reg_gamma))",
        margin=0mm
    )

    # 训练MSE直方图
    histogram!(p[1], log.(df.train_mse),
        bins=20,
        label="Train-MSE",
        xlabel=(idx == num_rows) ? "log(Train-MSE)" : "",
        ylabel="Frequency",
        color=:blue,
        alpha=0.6,
        xlims=xlims_train_mse)

    # 测试MSE直方图
    histogram!(p[2], log.(df.test_mse),
        bins=20,
        label="Test-MSE",
        xlabel=(idx == num_rows) ? "log(Test-MSE)" : "",
        ylabel="",
        color=:red,
        alpha=0.6,
        xlims=xlims_test_mse)

    histogram!(p[3], log.(df.final_reg_loss ./ df.init_reg_loss),
        bins=20,
        label="Reg Loss Ratio",
        xlabel=(idx == num_rows) ? "log(final_reg_loss / init_reg_loss)" : "",
        ylabel="",
        color=:green,
        alpha=0.6,
        xlims=xlims_reg_loss_ratio)
    push!(collected_plots_for_main_figure, p)
end

# ADDED: Create the main composite plot from all collected gamma plots
main_plot_figure = plot(
    collected_plots_for_main_figure...,
    layout=(7, 1), size=(1600, 1800), compact=true, margin=0mm,
    left_margin=20mm
)
savefig(main_plot_figure, "result/finetune/$(model_name)_reg_stats.png")
# This will create the plot object. To save it, you would use savefig(main_plot_figure, "path/to/figure.png")
stat_df = DataFrame(stats_list)
stat_df = stat_df[!, ["reg_gamma", "mean_train_mse", "mean_test_mse", "median_train_mse",
    "median_test_mse", "std_train_mse",
    "std_test_mse", "mean_init_reg_loss", "mean_final_reg_loss",
    "median_init_reg_loss", "median_final_reg_loss",
    "std_init_reg_loss", "std_final_reg_loss"]]
CSV.write("result/finetune/$(model_name)_reg_stats.csv", stat_df)
