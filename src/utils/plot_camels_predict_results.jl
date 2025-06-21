# 绘制直方图,比较k50-f和m50-f的预测结果
using CSV, DataFrames, Plots, Statistics


function plot_model_stats(stats_df, model_name)
    # Extract metrics
    nse_test = stats_df[!, Symbol("nse-test")]
    mnse_test = stats_df[!, Symbol("mnse-test")] 
    fhv_test = stats_df[!, Symbol("fhv-test")]
    
    # Constrain NSE and KGE between -1 and 1
    # nse_test[nse_test .< -1] .= -1
    # mnse_test[mnse_test .< -1] .= -1 
    # fhv_test[fhv_test .> 100] .= 100
    
    # Create bins
    bins = 0:0.05:1
    
    # Create subplot layout with increased top margin
    p = plot(layout=(1,3), size=(1200,400), top_margin=10Plots.mm, left_margin=5Plots.mm)
    
    # Calculate statistics
    nse_mean = mean(nse_test)
    nse_median = median(nse_test)
    mnse_mean = mean(mnse_test)
    mnse_median = median(mnse_test)
    fhv_mean = mean(fhv_test)
    fhv_median = median(fhv_test)

    # Define 3 color palettes
    palette1 = (:skyblue, :salmon, :mediumseagreen)  # Soft and professional

    # Plot NSE histogram
    histogram!(p[1], nse_test, bins=bins, label="NSE", color=palette1[1], tickfontsize=12, legendfontsize=12)
    vline!(p[1], [nse_mean], label="mean", color=:black, linewidth=2)
    vline!(p[1], [nse_median], label="median", color=:grey, linestyle=:dash, linewidth=2)
    annotate!(p[1], mean(xlims(p[1])), maximum(ylims(p[1]))*1.05, "mean=$(round(nse_mean,digits=3)), median=$(round(nse_median,digits=3))", fontsize=10)
    
    # Plot KGE histogram  
    histogram!(p[2], mnse_test, bins=bins, label="KGE", color=palette1[2], tickfontsize=12, legendfontsize=12)
    vline!(p[2], [mnse_mean], label="mean", color=:black, linewidth=2)
    vline!(p[2], [mnse_median], label="median", color=:grey, linestyle=:dash, linewidth=2)
    annotate!(p[2], mean(xlims(p[2])), maximum(ylims(p[2]))*1.05, "mean=$(round(mnse_mean,digits=3)), median=$(round(mnse_median,digits=3))", fontsize=10)
    
    # Plot FHV histogram
    histogram!(p[3], fhv_test, bins=0:2.5:100, label="FHV", color=palette1[3], tickfontsize=12, legendfontsize=12)
    vline!(p[3], [fhv_mean], label="mean", color=:black, linewidth=2)
    vline!(p[3], [fhv_median], label="median", color=:grey, linestyle=:dash, linewidth=2)
    annotate!(p[3], mean(xlims(p[3])), maximum(ylims(p[3]))*1.05, "mean=$(round(fhv_mean,digits=3)), median=$(round(fhv_median,digits=3))", fontsize=10)
    
    # Set y-axis label for leftmost plot with adjusted position
    ylabel!(p[1], model_name, labelfontsize=16, left_margin=10Plots.mm)
    return p
end

# model_name_list = ["k50_base(step)"] # "m50-p", "m50-f", "d-hbv", "k50-f", "k50-p", "hbv" "exphydro(cont2,withst)"
# for model_name in model_name_list
#     m50_predict_stats = CSV.read("src/result/stats/$model_name-criteria.csv", DataFrame)
#     tmp_plot = plot_model_stats(m50_predict_stats, model_name)
#     savefig(tmp_plot, "src/plots/$model_name-hist.png")
# end
model_name = "k50_base"
predict_stats = CSV.read("result/v2/$model_name/criteria.csv", DataFrame)
tmp_plot = plot_model_stats(predict_stats, model_name)
savefig(tmp_plot, "result/v2/plots/$model_name-hist.png")