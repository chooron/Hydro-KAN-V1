using Plots
using CSV, DataFrames, JLD2, Dates
using Colors

basemodel_dir = "result/exphydro(516)"
basin_id = "01013500"
#* load data
camelsus_cache = load("data/camelsus/$(basin_id).jld2")
data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
test_x, test_y, test_timepoints = camelsus_cache["test_x"], camelsus_cache["test_y"], camelsus_cache["test_timepoints"]
train_daterange = Date(1980, 10, 1):Day(1):Date(2000, 9, 30)
test_daterange = Date(2000, 10, 1):Day(1):Date(2010, 9, 30)

logocolors = Colors.JULIA_LOGO_COLORS
p = plot(train_daterange, train_y, legend=true, axis=true, grid=false, color=logocolors.blue, tick_labels=false, ylabel="Streamflow (mm/d)",
linewidth=3, label="Train", fontfamily="Times", tickfontsize=18, legendfontsize=18, fontsize=18, dpi=600, right_margin=10Plots.mm, labelfontsize=18,
xticks=([Date(1980,10,1), Date(2000,10,1), Date(2010,9,30)], ["1980-10-01", "2000-10-01", "2010-09-30"]))
plot!(test_daterange, test_y, legend=true, axis=true, grid=false, color=logocolors.red, tick_labels=false,
linewidth=3, label="Test", fontfamily="Times", tickfontsize=18, legendfontsize=18, fontsize=18)

savefig(p, "src/v2/tmp/figures/streamflow.png")
