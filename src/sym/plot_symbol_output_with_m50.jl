using CSV, DataFrames, JLD2, Dates
using Plots
include("../utils/criteria.jl")

function plot_figure(basin_id, show_ylabel=true)
    symbol_output_df = CSV.read("src/sym/output/$(basin_id)_output.csv", DataFrame)
    k50_output = load("result/k50_reg(1e-2)/$(basin_id)/predicted_df.jld2")["y_reg_pred"]
    exphydro_output_df = CSV.read("result/exphydro(516)/$(basin_id)/model_outputs.csv", DataFrame)
    test_len = CSV.read("result/exphydro(516)/$(basin_id)/test_predicted_df.csv", DataFrame) |> nrow
    m50_df = CSV.read("result/m50_base/$(basin_id)/test_predicted_df.csv", DataFrame)
    total_len = nrow(symbol_output_df)
    # 准备数据
    obs = symbol_output_df.qreal[total_len-test_len+1:end]
    pred_symbolize = symbol_output_df.flow[total_len-test_len+1:end]
    pred_exphydro = exphydro_output_df.qsim[total_len-test_len+1:end]
    pred_k50 = k50_output[total_len-test_len+1:end]
    pred_m50 = m50_df.pred

    nse_symbolize, mnse_symbolize, fhv_symbolize = nse(obs, pred_symbolize), mnse(obs, pred_symbolize), fhv(obs, pred_symbolize)
    nse_exphydro, mnse_exphydro, fhv_exphydro = nse(obs, pred_exphydro), mnse(obs, pred_exphydro), fhv(obs, pred_exphydro)
    nse_k50, mnse_k50, fhv_k50 = nse(obs, pred_k50), mnse(obs, pred_k50), fhv(obs, pred_k50)
    nse_m50, mnse_m50, fhv_m50 = nse(obs, pred_m50), mnse(obs, pred_m50), fhv(obs, pred_m50)
    basin_id = basin_id[1]=='0' ? basin_id[2:end] : basin_id
    dates = Date(2000, 10, 1):Day(1):Date(2010, 9, 30)
    # 创建子图
    p1 = plot(dates, [obs pred_symbolize], 
        label=["Observed($basin_id)" "K50(Symbolize)"],
        color=[:black :skyblue],
        linestyle=[:solid :solid],
        linewidth=[1 2],
        alpha=[1.0 0.8],
        fontfamily="Times",
        xlabel="Date",
        ylabel=show_ylabel ? "Streamflow (mm/d)" : "",
        tickfont=font(12, "Times"),
        guidefont=font(14, "Times"),
        titlefont=font(16, "Times"),
        size=(500, 300),
        dpi=300)
    plot!(p1, legend=:topright)

    annote_pos1 = (Date(2004, 1, 1), 0.4*max(maximum(obs), maximum(pred_symbolize)))
    annotate!(p1, [(annote_pos1..., Plots.text("NSE = $(round(nse_symbolize, digits=2))\nmNSE = $(round(mnse_symbolize, digits=2))\nFHV = $(round(fhv_symbolize, digits=2))", 14, "Times", :right, :bottom, color=:black))])

    p2 = plot(dates, [obs pred_exphydro],
        label=["Observed($basin_id)" "Exp-Hydro"],
        color=[:black :salmon],
        linestyle=[:solid :solid], 
        linewidth=[1 2],
        alpha=[1.0 0.8],
        fontfamily="Times",
        ylabel=show_ylabel ? "Streamflow (mm/d)" : "",
        tickfont=font(12, "Times"),
        guidefont=font(14, "Times"),
        titlefont=font(16, "Times"),
        size=(500, 300),
        dpi=300)
    plot!(p2, legend=:topright)

    annote_pos2 = (Date(2004, 1, 1), 0.4*max(maximum(obs), maximum(pred_exphydro)))
    annotate!(p2, [(annote_pos2..., Plots.text("NSE = $(round(nse_exphydro, digits=2))\nmNSE = $(round(mnse_exphydro, digits=2))\nFHV = $(round(fhv_exphydro, digits=2))", 14, "Times", :right, :bottom, color=:black))])

    p3 = plot(dates, [obs pred_k50],
        label=["Observed($basin_id)" "K50"],
        color=[:black :mediumseagreen],
        linestyle=[:solid :solid],
        linewidth=[1 2], 
        alpha=[1.0 0.8],
        fontfamily="Times",
        ylabel=show_ylabel ? "Streamflow (mm/d)" : "",
        tickfont=font(12, "Times"),
        guidefont=font(14, "Times"),
        titlefont=font(16, "Times"),
        size=(500, 300),
        dpi=300)
        
    plot!(p3, legend=:topright)

    annote_pos3 = (Date(2004, 1, 1), 0.4*max(maximum(obs), maximum(pred_k50)))
    annotate!(p3, [(annote_pos3..., Plots.text("NSE = $(round(nse_k50, digits=2))\nmNSE = $(round(mnse_k50, digits=2))\nFHV = $(round(fhv_k50, digits=2))", 14, "Times", :right, :bottom, color=:black))])
        
    p4 = plot(dates, [obs pred_m50],
        label=["Observed($basin_id)" "M50"],
        color=[:black :mediumpurple],
        linestyle=[:solid :solid],
        linewidth=[1 2], 
        alpha=[1.0 0.8],
        fontfamily="Times",
        ylabel=show_ylabel ? "Streamflow (mm/d)" : "",
        tickfont=font(12, "Times"),
        guidefont=font(14, "Times"),
        titlefont=font(16, "Times"),
        size=(500, 300),
        dpi=300)
        
    plot!(p4, legend=:topright)

    annote_pos4 = (Date(2004, 1, 1), 0.4*max(maximum(obs), maximum(pred_m50)))
    annotate!(p4, [(annote_pos4..., Plots.text("NSE = $(round(nse_m50, digits=2))\nmNSE = $(round(mnse_m50, digits=2))\nFHV = $(round(fhv_m50, digits=2))", 14, "Times", :right, :bottom, color=:black))])


    # 组合子图
    combined_plot = plot(p2, p4, p3, p1, 
        layout=(4,1),
        size=(500, 900),
        legend=false,
        framestyle=:box,
        margin=0.1Plots.mm)
    combined_plot
end


p1 = plot_figure("01137500", true)
p2 = plot_figure("02053200", false)
p3 = plot_figure("03281500", false)

p = plot(p1, p2, p3, layout=(1,3), size=(1500, 900), bottom_margin=6Plots.mm, left_margin=6Plots.mm, dpi=300)
savefig(p, "src/plots/figures/symbol_output_with_m50.png")