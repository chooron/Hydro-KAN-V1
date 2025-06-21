using CSV, DataFrames, Dates, Plots, Statistics
include("../utils/criteria.jl")

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

function stat_predict_results(model_name)
    base_path = "result/v2/$model_name"
    save_path = "src/v2/stats"
    # Get all subdirectories in m50 folder
    subdirs = filter(isdir, readdir(base_path, join = true))
    criteria_all = []

    # Read predict_df.csv from each subdir if it exists
    for dir in subdirs
        train_pred_file = joinpath(dir, "train_predicted_df.csv")
        test_pred_file = joinpath(dir, "test_predicted_df.csv")
        station_id = basename(dir)
        train_df = CSV.read(train_pred_file, DataFrame)
        test_df = CSV.read(test_pred_file, DataFrame)

        # Split into train and test periods
        train_pred, train_obs = train_df[!, :val_pred], train_df[!, :obs]
        test_pred, test_obs = test_df[!, :val_pred], test_df[!, :obs]

        criteria_dict = Dict(
            "station_id" => station_id,
            "rmse-train" => rmse(train_obs, train_pred), "rmse-test" => rmse(test_obs, test_pred),
            "mae-train" => mae(train_obs, train_pred), "mae-test" => mae(test_obs, test_pred),
            "nse-train" => nse(train_obs, train_pred), "nse-test" => nse(test_obs, test_pred),
            "mnse-train" => mnse(train_obs, train_pred), "mnse-test" => mnse(test_obs, test_pred),
            "fhv-train" => fhv(train_obs, train_pred, h=0.01), "fhv-test" => fhv(test_obs, test_pred, h=0.01),
            "kge-train" => kge(train_obs, train_pred), "kge-test" => kge(test_obs, test_pred),
        )
        push!(criteria_all, criteria_dict)
    end

    criteria_df = DataFrame(criteria_all)
    criteria_df_name = sort(filter(x -> x != "station_id", names(criteria_df)), rev=true)
    criteria_df = criteria_df[!, ["station_id", criteria_df_name...]]
    CSV.write("$save_path/$model_name-criteria.csv", criteria_df)

    return criteria_df
end

model_name = "k50_base"
stats_df = stat_predict_results(model_name)
stats_fig = plot_model_stats(stats_df, model_name)
display(stats_fig)
