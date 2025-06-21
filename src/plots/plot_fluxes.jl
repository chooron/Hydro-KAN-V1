using Plots
using CSV, DataFrames

basemodel_dir = "result/exphydro(516)"
basin_id = "01013500"
exphydro_df = CSV.read("$(basemodel_dir)/$(basin_id)/model_outputs.csv", DataFrame)
snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
et_vec, flow_vec = exphydro_df[!, "et"], exphydro_df[!, "qsim"]
pr_vec, melt_vec = exphydro_df[!, "pr"], exphydro_df[!, "melt"]
S0_vec, S1_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
logocolors = Colors.JULIA_LOGO_COLORS
[logocolors.blue, logocolors.red, logocolors.green, logocolors.purple]
# 创建2*3的组合图
p = plot(
    plot(et_vec[1:1000], legend=false, axis=true,  grid=false, color=logocolors.blue, tick_labels=false, xticks=false, yticks=false,
    linewidth=3, label="ET", fontfamily="Times", tickfontsize=16, legendfontsize=18, fontsize=18),
    plot(flow_vec[1:1000], legend=false, axis=true, grid=false, color=logocolors.red, tick_labels=false, xticks=false, yticks=false,
     linewidth=3, label="Q", fontfamily="Times", tickfontsize=16, legendfontsize=18, fontsize=18),
    plot(pr_vec[1:1000], legend=false, axis=true, grid=false, color=logocolors.green, tick_labels=false, xticks=false, yticks=false,
     linewidth=3, label="Pr", fontfamily="Times", tickfontsize=16, legendfontsize=18, fontsize=18),
    plot(melt_vec[1:1000], legend=false, axis=true, grid=false, color=logocolors.blue, tick_labels=false, xticks=false, yticks=false,
    linewidth=3, label="M", fontfamily="Times", tickfontsize=16, legendfontsize=18, fontsize=18),
    plot(S0_vec[1:1000], legend=false, axis=true, grid=false, color=logocolors.red, tick_labels=false, xticks=false, yticks=false,
    linewidth=3, label="S0", fontfamily="Times", tickfontsize=16, legendfontsize=18, fontsize=18),
    plot(S1_vec[1:1000], legend=false, axis=true, grid=false, color=logocolors.green, tick_labels=false, xticks=false, yticks=false,
    linewidth=3, label="S1", fontfamily="Times", tickfontsize=16, legendfontsize=18, fontsize=18),
    layout=(2,3),
    dpi=600,
    rightmargin=3Plots.mm,
    size=(800, 400)
)
savefig(p, "src/v2/tmp/figures/fluxes.png")
