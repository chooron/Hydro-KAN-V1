using Plots, Measures
using CSV, DataFrames, JLD2, Dates
using Colors
using Statistics

basemodel_dir = "result/exphydro(516)"
basin_id = "06191500"

function plot_streamflow1(basin_id, color, show_ylabel1, show_ylabel2, show_xlabel)
    #* load data
    camelsus_cache = load("data/camelsus/$(basin_id).jld2")
    data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
    total_daterange = Date(1980, 10, 1):Day(1):Date(2010, 9, 30)
    temp_df = DataFrame(Date=total_daterange, Streamflow=data_y, Prcp=data_x[:, 2])


    # 添加年份列
    temp_df[!, :Year] = year.(temp_df.Date)
    temp_df[!, :DayOfYear] = dayofyear.(temp_df.Date)

    # 选择特定年份进行绘图
    selected_years = [1992] # 您可以选择其他年份
    # multi_year_df = filter(row -> row.Year in selected_years, temp_df)
    multi_year_df = temp_df[(temp_df.Date .< Date(1992, 5, 30)) .& (temp_df.Date .> Date(1992, 3, 1)), :]

    # 创建绘图
    p = plot(multi_year_df.Date, multi_year_df.Streamflow,
        label="Streamflow",
        xlabel=show_xlabel ? "Date" : "",
        ylabel=show_ylabel1 ? "Streamflow (mm/d)" : "",
        fontfamily="Times",
        lw=2,
        legend=false,
        dpi=300,
        xlabelfontsize=16,
        ylabelfontsize=16,
        xtickfontsize=16,
        ytickfontsize=16,
        size=(400, 300),
        # xticks=([Date(1992, 1, 1), Date(1992, 6, 1), Date(1992, 12, 1)], ["1992/1", "1992/6", "1992/12"]),
        xticks=([Date(1992, 3, 1), Date(1992, 5, 1)], ["1992/3", "1992/5"]),
        ylims=(0, 2 * maximum(multi_year_df.Streamflow)),
        color=color)  # 设置5个均匀分布的刻度

    # 添加第二个Y轴用于降水，并绘制柱状图
    plot!(twinx(), multi_year_df.Date, multi_year_df.Prcp,
        seriestype=:bar,
        label="Precipitation",
        ylabel=show_ylabel2 ? "Precipitation (mm/d)" : "",
        color=logocolors.blue,
        alpha=0.8,
        ylabelfontsize=18,
        ytickfontsize=16,
        yflip=true,  # 反转Y轴
        legend=false, # 禁用降雨图的图例
        bar_width=2.0, # 可以调整柱子的宽度
        linecolor=:match, # 移除柱状图边框
        frame_style=:box,
        ylims=(0, 2.0 * maximum(multi_year_df.Prcp)), # 设置降雨Y轴范围
)  # 设置5个均匀分布的刻度

    # 保存图像
    # savefig(p, "src/v2/tmp/figures/$(basin_id)_rainfall_runoff.png")
    return p
end

function plot_streamflow2(basin_id, color, show_ylabel1, show_ylabel2, show_xlabel)
    #* load data
    camelsus_cache = load("data/camelsus/$(basin_id).jld2")
    data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
    total_daterange = Date(1980, 10, 1):Day(1):Date(2010, 9, 30)
    temp_df = DataFrame(Date=total_daterange, Streamflow=data_y, Prcp=data_x[:, 2])


    # 添加年份列
    temp_df[!, :Year] = year.(temp_df.Date)
    temp_df[!, :DayOfYear] = dayofyear.(temp_df.Date)

    # 选择特定年份进行绘图
    selected_years = [1992] # 您可以选择其他年份
    # multi_year_df = filter(row -> row.Year in selected_years, temp_df)
    multi_year_df = temp_df[(temp_df.Date .< Date(1992, 5, 30)) .& (temp_df.Date .> Date(1992, 3, 1)), :]

    # 创建绘图
    p = plot(multi_year_df.Date, multi_year_df.Streamflow,
        label="Streamflow",
        xlabel=show_xlabel ? "Date" : "",
        ylabel=show_ylabel1 ? "Streamflow (mm/d)" : "",
        fontfamily="Times",
        lw=2,
        legend=false,
        dpi=300,
        xlabelfontsize=16,
        ylabelfontsize=16,
        xtickfontsize=16,
        ytickfontsize=16,
        size=(400, 300),
        # xticks=([Date(1992, 1, 1), Date(1992, 6, 1), Date(1992, 12, 1)], ["1992/1", "1992/6", "1992/12"]),
        xticks=([Date(1992, 3, 1), Date(1992, 5, 1)], ["1992/3", "1992/5"]),
        ylims=(0, 2 * maximum(multi_year_df.Streamflow)),
        color=color)  # 设置5个均匀分布的刻度

    # 添加第二个Y轴用于降水，并绘制柱状图
    plot!(twinx(), multi_year_df.Date, multi_year_df.Prcp,
        seriestype=:bar,
        label="Precipitation",
        ylabel=show_ylabel2 ? "Precipitation (mm/d)" : "",
        color=logocolors.blue,
        alpha=0.8,
        ylabelfontsize=18,
        ytickfontsize=16,
        yflip=true,  # 反转Y轴
        legend=false, # 禁用降雨图的图例
        bar_width=2.0, # 可以调整柱子的宽度
        linecolor=:match, # 移除柱状图边框
        frame_style=:box,
        ylims=(0, 2.0 * maximum(multi_year_df.Prcp)), # 设置降雨Y轴范围
)  # 设置5个均匀分布的刻度

    # 保存图像
    # savefig(p, "src/v2/tmp/figures/$(basin_id)_rainfall_runoff.png")
    return p
end

logocolors = Colors.JULIA_LOGO_COLORS
p1 = plot_streamflow1("01137500", logocolors.red, true, true, false)
p2 = plot_streamflow2("02053200", logocolors.red, true, true, false)

savefig(p1, "src/v2/tmp/figures/01137500.png")
savefig(p2, "src/v2/tmp/figures/02053200.png")
# p2 = plot_streamflow("03281500", logocolors.green, true, true)
# p3 = plot_streamflow("06191500", logocolors.purple, true, true)

