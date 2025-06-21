# 分析K50,M50与Exp-Hydro模型的差别
using DataFrames, CSV
using Plots, StatsPlots
using Statistics

m50_criteria = CSV.read("src/stats/m50_base-criteria.csv", DataFrame)
k50_criteria = CSV.read("src/stats/k50_base-criteria.csv", DataFrame)
exphydro_criteria = CSV.read("src/stats/exphydro(disc,withst)-criteria.csv", DataFrame)

m50_test_nse = m50_criteria[!, "nse-test"]
k50_test_nse = k50_criteria[!, "nse-test"]
exphydro_test_nse = exphydro_criteria[!, "nse-test"]

m50_test_nse[m50_test_nse .< -1] .= -1
k50_test_nse[k50_test_nse .< -1] .= -1
exphydro_test_nse[exphydro_test_nse .< -1] .= -1

# 整理成“长格式”
groups = vcat(repeat(["M50"], length(m50_test_nse)), 
              repeat(["K50"], length(k50_test_nse)))
values = vcat(m50_test_nse, k50_test_nse)
color_values = vcat(exphydro_test_nse, exphydro_test_nse)


# --- 2. 绘制 "雨云图" ---

# 创建一个组合图，这里我们直接在 `violin` 和 `scatter` 中使用 `group` 参数
# StatsPlots 会自动处理分组和位置
plot(
    # Part 1: "云" - 提琴图
    violin(groups, values, 
           side=:right, # 让提琴图只显示在右半边，为散点留出空间
           legend=false,
           linewidth=0, # 不画边框
           color=[:skyblue :orange],
           alpha=0.6), # 设为半透明

    # Part 2: "雨" - 带抖动的散点
    scatter(groups, values,
            marker_z = color_values,      # 按 exphydro 值上色
            color = :viridis,             # 使用 viridis 颜色梯度
            colorbar_title = "ExpHydro NSE",
            markersize = 4,
            markerstrokewidth = 0.5,
            markeralpha = 0.8,
            jitter = 0.3,                 # 直接使用 StatsPlots 的 jitter 功能
            label=""),

    # 美化
    title = "Model Performance (Raincloud Plot)",
    ylabel = "NSE (Test)",
    xlabel = "Model",
    ylims = (0, 1)
)
