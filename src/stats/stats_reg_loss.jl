using JLD2
using DataFrames
using Plots
using DelimitedFiles
using Statistics


function main(reg_gamma)
    basin_file = readdlm(joinpath("data/basin_ids/basins_all.txt"), ',')
    basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")
    gamma_dict = Dict("1e-2" => 1e-2, "1e-3" => 1e-3, "5e-3" => 5e-3)
    data_list = []
    for basin_id in basins_available
        result_path = "result/k50_reg($reg_gamma)/$basin_id/loss_df.jld2"
        result = load(result_path)

        adam_loss_sorted = sort(result["adam_loss_df"], :train_loss)
        adam_best_train_loss = adam_loss_sorted[1, :train_loss]
        adam_best_val_loss = adam_loss_sorted[1, :val_loss]
        lbfgs_loss = result["lbfgs_loss_df"]
        init_reg_loss = lbfgs_loss[1, :reg_loss] ./ gamma_dict[reg_gamma]
        lbfgs_loss_sorted = sort(result["lbfgs_loss_df"], :val_loss)
        lbfgs_best_train_loss = lbfgs_loss_sorted[1, :train_loss]
        lbfgs_best_val_loss = lbfgs_loss_sorted[1, :val_loss]
        lbfgs_best_reg_loss = lbfgs_loss_sorted[1, :reg_loss] ./ gamma_dict[reg_gamma]


        tmp_loss_list = [basin_id, adam_best_train_loss, adam_best_val_loss, init_reg_loss, lbfgs_best_train_loss, lbfgs_best_val_loss, lbfgs_best_reg_loss]
        push!(data_list, NamedTuple{(:basin_id, :adam_train_loss, :adam_val_loss, :init_reg_loss, :lbfgs_train_loss, :lbfgs_val_loss, :lbfgs_reg_loss)}(tmp_loss_list))
    end

    data_df = DataFrame(data_list, [:basin_id, :adam_train_loss, :adam_val_loss, :init_reg_loss, :lbfgs_train_loss, :lbfgs_val_loss, :lbfgs_reg_loss])
    CSV.write("src/stats/k50_reg($reg_gamma)_loss_df.csv", data_df)
    @info "lbfgs_train_loss: $(median(data_df.lbfgs_train_loss)), lbfgs_val_loss: $(median(data_df.lbfgs_val_loss)), lbfgs_reg_loss: $(median(data_df.lbfgs_reg_loss))"
    @info "adam_train_loss: $(median(data_df.adam_train_loss)), adam_val_loss: $(median(data_df.adam_val_loss)), init_reg_loss: $(median(data_df.init_reg_loss))"

    fig1 = histogram(data_df[!, :init_reg_loss], bins=20, label="", alpha=0.8, tickfontsize=16, xticks=[0, 100, 200, 300], color=:skyblue)
    histogram!(data_df[!, :lbfgs_reg_loss], bins=20, label="", alpha=0.8, tickfontsize=16, xticks=[0, 100, 200, 300], color=:salmon)

    fig2 = histogram(filter(x -> x < 1, data_df[!, :adam_train_loss]), bins=20, label="", alpha=0.8, tickfontsize=16, xticks=[0, 0.5, 1], color=:skyblue)
    histogram!(filter(x -> x < 1, data_df[!, :lbfgs_train_loss]), bins=20, label="", alpha=0.8, tickfontsize=16, xticks=[0, 0.5, 1], color=:salmon)

    fig3 = histogram(filter(x -> x < 1, data_df[!, :adam_val_loss]), bins=20, label="", alpha=0.8, tickfontsize=16, xticks=[0, 0.5, 1], color=:skyblue)
    histogram!(filter(x -> x < 1, data_df[!, :lbfgs_val_loss]), bins=20, label="", alpha=0.8, tickfontsize=16, xticks=[0, 0.5, 1], color=:salmon)

    fig = plot(
        fig1, fig2, fig3, layout=(1, 3), size=(750, 250), fontfamily="Times",
        bottom_margin=0.2Plots.mm, left_margin=0.2Plots.mm, dpi=300,
    )
    return fig
end

fig1 = main("1e-2")
fig2 = main("5e-3")
fig3 = main("1e-3")
fig = plot(fig1, fig2, fig3, layout=(3, 1), size=(750, 750), dpi=300, framestyle=:box, fontcolor=:black)

savefig(fig, "src/plots/figures/k50_reg_loss_hist.png")
