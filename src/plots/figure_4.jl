# 绘制直方图,比较k50-f和m50-f的预测结果
using CSV, DataFrames, Plots, Statistics


function plot_model_stats(stats_df, model_name, last=false)
    # Filter out rows where nse-test < -1
    # stats_df = stats_df[stats_df[!, Symbol("nse-test")] .>= -1, :]

    # Extract metrics
    nse_test = stats_df[!, Symbol("nse-test")]
    mnse_test = stats_df[!, Symbol("mnse-test")]
    rmse_test = stats_df[!, Symbol("rmse-test")]
    fhv_test = stats_df[!, Symbol("fhv-test")]

    # Calculate statistics
    nse_mean = mean(nse_test)
    nse_median = median(nse_test)
    mnse_mean = mean(mnse_test)
    mnse_median = median(mnse_test)
    fhv_mean = mean(fhv_test)
    fhv_median = median(fhv_test)
    rmse_mean = mean(rmse_test)
    rmse_median = median(rmse_test)

    # Constrain NSE and KGE between -1 and 1
    nse_test[nse_test.<0] .= 0
    mnse_test[mnse_test.<0] .= 0
    fhv_test[fhv_test.>75] .= 75
    rmse_test[rmse_test.>10] .= 10

    # Create bins
    bins = 0:0.04:1

    # Create subplot layout with increased top margin and reduced spacing
    p = plot(
        layout=(1, 4), size=(2000, 300), top_margin=6Plots.mm, left_margin=5Plots.mm, framestyle=:box,
        right_margin=5Plots.mm, dpi=300, margin=0.1 * Plots.mm, bottom_margin=2Plots.mm
    )

    # Define 3 color palettes
    palette1 = (:skyblue, :salmon, :darkseagreen2, :thistle)  # Soft and professional

    # Plot NSE histogram
    histogram!(p[1], nse_test, bins=bins, label="NSE", color=palette1[1], tickfontsize=12, legendfontsize=12, fontfamily="Times", legend=false)
    vline!(p[1], [nse_mean], label="mean", color=:black, linewidth=2, fontfamily="Times")
    vline!(p[1], [nse_median], label="median", color=:grey, linestyle=:dash, linewidth=2, fontfamily="Times")
    annotate!(p[1], mean(xlims(p[1])), maximum(ylims(p[1])) * 1.08, "mean=$(round(nse_mean,digits=3)), median=$(round(nse_median,digits=3))", fontsize=12, fontfamily="Times")

    # Plot KGE histogram  
    histogram!(p[2], mnse_test, bins=bins, label="mNSE", color=palette1[2], tickfontsize=12, legendfontsize=12, fontfamily="Times", legend=false)
    vline!(p[2], [mnse_mean], label="mean", color=:black, linewidth=2, fontfamily="Times")
    vline!(p[2], [mnse_median], label="median", color=:grey, linestyle=:dash, linewidth=2, fontfamily="Times")
    annotate!(p[2], mean(xlims(p[2])), maximum(ylims(p[2])) * 1.08, "mean=$(round(mnse_mean,digits=3)), median=$(round(mnse_median,digits=3))", fontsize=12, fontfamily="Times")

    # Plot FHV histogram
    histogram!(p[3], rmse_test, bins=0:0.5:10, label="RMSE", color=palette1[3], tickfontsize=12, legendfontsize=12, fontfamily="Times", legend=false)
    vline!(p[3], [rmse_mean], label="mean", color=:black, linewidth=2, fontfamily="Times")
    vline!(p[3], [rmse_median], label="median", color=:grey, linestyle=:dash, linewidth=2, fontfamily="Times")
    annotate!(p[3], mean(xlims(p[3])), maximum(ylims(p[3])) * 1.08, "mean=$(round(rmse_mean,digits=3)), median=$(round(rmse_median,digits=3))", fontsize=12, fontfamily="Times")

    # Plot FHV histogram
    histogram!(p[4], fhv_test, bins=0:2:75, label="FHV", color=palette1[4], tickfontsize=12, legendfontsize=12, fontfamily="Times", legend=false)
    vline!(p[4], [fhv_mean], label="mean", color=:black, linewidth=2, fontfamily="Times")
    vline!(p[4], [fhv_median], label="median", color=:grey, linestyle=:dash, linewidth=2, fontfamily="Times")
    annotate!(p[4], mean(xlims(p[4])), maximum(ylims(p[4])) * 1.08, "mean=$(round(fhv_mean,digits=3)), median=$(round(fhv_median,digits=3))", fontsize=12, fontfamily="Times")

    # Set y-axis label for leftmost plot with adjusted position
    ylabel!(p[1], model_name, labelfontsize=14, right_margin=5Plots.mm)

    if last
        xlabel!(p[1], "NSE", fontsize=12, fontfamily="Times")
        xlabel!(p[2], "mNSE", fontsize=12, fontfamily="Times")
        xlabel!(p[3], "RMSE", fontsize=12, fontfamily="Times")
        xlabel!(p[4], "FHV", fontsize=12, fontfamily="Times")
    end
    return p
end

model_name_list = ["exphydro(disc,withst)", "m50_base", "k50_base"] # "m50-p", "m50-f", "d-hbv", "k50-f", "k50-p", "hbv" "exphydro(cont2,withst)"
show_model_name = ["Exp-Hydro", "M50", "K50"]
plot_list = []
show_xlabel = [false, false, true]
for (model_name, show_name, show_x) in zip(model_name_list, show_model_name, show_xlabel)
    m50_predict_stats = CSV.read("result/stats/$model_name-criteria.csv", DataFrame)
    tmp_plot = plot_model_stats(m50_predict_stats, show_name, show_x)
    push!(plot_list, tmp_plot)
    savefig(tmp_plot, "src/v2/plots/figures/$(model_name)-hist.png")
end

plot(plot_list..., layout=grid(length(model_name_list), 1), size=(1200, 250 * length(model_name_list)), dpi=600)
savefig("src/v2/plots/figures/combined_hist.png")