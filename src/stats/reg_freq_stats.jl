#= 
* 筛选条件
*- 1. lbfgs_train_loss < adam_train_loss
*- 2. lbfgs_reg_loss < 0.5 * init_reg_loss
=#
using CSV, DataFrames

function filter_sample(reg_gamma::String)
    model_name = "k50_reg($reg_gamma)"
    base_path = "src/stats/$(model_name)_loss_df.csv"

    # 读取CSV文件
    df = CSV.read(base_path, DataFrame)

    # 应用筛选条件
    # 条件1: lbfgs_train_loss < adam_train_loss
    condition1 = df.lbfgs_train_loss .< df.adam_train_loss

    # 条件2: lbfgs_reg_loss < 0.5 * init_reg_loss
    condition2 = df.lbfgs_reg_loss .< (0.2 .* df.init_reg_loss)

    # 合并筛选条件
    filtered_df = df[condition2, :]

    # 创建统计表
    stats = DataFrame(
        threshold=Float64[],
        count=Int[],
        percentage=Float64[]
    )

    # 统计不同阈值下的结果
    for threshold in 0.1:0.01:1.0
        condition = df.lbfgs_reg_loss .< (threshold .* df.init_reg_loss)
        count = sum(condition)
        percentage = round(count / nrow(df) * 100, digits=2)

        push!(stats, (threshold, count, percentage))
    end
    return stats
end

gamma_df_1 = filter_sample("1e-2")
gamma_df_2 = filter_sample("5e-3")
gamma_df_3 = filter_sample("1e-3")


plot(gamma_df_1.threshold, gamma_df_1.percentage, label="1e-2",
    labelfontsize=14, tickfontsize=12, legendfontsize=12,
    xlabel="Threshold", ylabel="Percentage", fontfamily="Times", dpi=300)
plot!(gamma_df_2.threshold, gamma_df_2.percentage, label="5e-3")
plot!(gamma_df_3.threshold, gamma_df_3.percentage, label="1e-3")

# savefig("src/plots/figures/k50_reg_threshold_stats.png")
