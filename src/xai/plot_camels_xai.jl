using CSV, DataFrames, Dates
using Plots, Statistics, JLD2, Shapefile, Measures

shp_path = "data/gis/us-main.shp"
station_position = CSV.read("data/gis/gauge_info.csv", DataFrame)

function plot_camels_xai(method, model, var, colorbar)
    xai_result = CSV.read("src/xai/result/camels_$(method)_values_$(model).csv", DataFrame)
    # 筛选出只在station_xai_result出现的station_position
    select_position = station_position[in.(station_position[!, :GAGE_ID], Ref(xai_result[!, :basin_id])), :]
    station_lat = select_position[!, :LAT]
    station_lon = select_position[!, :LON]
    xai_total = sum(xai_result[!, [:S0, :S1, :Melt, :Rainfall]] |> Array, dims=2)
    xai_result = xai_result[!, var] ./ xai_total

    table = Shapefile.Table(shp_path)
    shapes = Shapefile.shapes(table)

    # 创建绘图
    p1 = plot(xlabel="longitude", ylabel="latitude", aspect_ratio=:equal, legend=false,
        xticks=nothing, yticks=nothing, grid=false, framestyle=:none,
        dpi=300, fontfamily="Times")

    # 绘制 SHP 文件中的地理边界
    for shape in shapes
        plot!(p1, shape, fillalpha=0.5, linecolor=:darkgrey, fillcolor=:lightgrey, linewidth=0.5)
    end

    # 绘制站点位置
    scatter!(p1, station_lon, station_lat,
        marker_z=xai_result,
        color=colorbar,
        markersize=3,
        margin=0.1Plots.mm,
        markerstrokewidth=0.1,
        markerstrokecolor=:gray,
        colorbar=false,
        colorbar_tickfontsize=14,
        clims=(0.0, 1.0))
    return p1
end
model = "k50_reg(1e-2)"
method = "ig"
map(zip(["S0", "S1", "Melt", "Rainfall"], [:Reds, :Greens, :Purples, :Blues])) do (var, colorbar)
    fig = plot_camels_xai(method, model, var, colorbar)
    savefig(fig, "src/xai/plots/camels_$(method)_$(model)_$(var)_map.png")
end

