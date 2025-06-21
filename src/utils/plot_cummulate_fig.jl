# 绘制qnn训练前后变化的预测性能变化
using Plots
using Statistics
using DataFrames, CSV, DelimitedFiles

m50_posttrain_result = CSV.read("src/result/stats/m50_f_d-criteria.csv", DataFrame)
k50_posttrain_result = CSV.read("src/result/stats/k50_f_d-criteria.csv", DataFrame)

function plot_cummulate_fig(posttrain_result, criteria)
	m50_posttrain_result, k50_posttrain_result = posttrain_result

	m50_posttrain_train_nse = m50_posttrain_result[!, "$criteria-train"]
	k50_posttrain_train_nse = k50_posttrain_result[!, "$criteria-train"]

	m50_posttrain_test_nse = m50_posttrain_result[!, "$criteria-test"]
	k50_posttrain_test_nse = k50_posttrain_result[!, "$criteria-test"]

	# Calculate median values
	m50_train_median = median(m50_posttrain_train_nse)
	k50_train_median = median(k50_posttrain_train_nse)
	m50_test_median = median(m50_posttrain_test_nse)
	k50_test_median = median(k50_posttrain_test_nse)

	m50_posttrain_train_nse[m50_posttrain_train_nse.<0] .= NaN
	k50_posttrain_train_nse[k50_posttrain_train_nse.<0] .= NaN

	m50_posttrain_test_nse[m50_posttrain_test_nse.<0] .= NaN
	k50_posttrain_test_nse[k50_posttrain_test_nse.<0] .= NaN

	sorted_m50_posttrain_train = sort(m50_posttrain_train_nse)
	sorted_k50_posttrain_train = sort(k50_posttrain_train_nse)

	sorted_m50_posttrain_test = sort(m50_posttrain_test_nse)
	sorted_k50_posttrain_test = sort(k50_posttrain_test_nse)

	# Calculate cumulative probabilities
	n = length(m50_posttrain_train_nse)
	cum_prob_train = collect(1:n) ./ n
	cum_prob_test = collect(1:n) ./ n


	# Create cumulative distribution plot
	p = scatter(sorted_m50_posttrain_train, cum_prob_train, label = "M50-train",
		marker = :circle, markersize = 3, markerstrokewidth = 0.2, markerstrokecolor = :black,
		alpha = 0.8, dpi = 300, size = (600, 400)
	)

	scatter!(
		sorted_k50_posttrain_train, cum_prob_train, label = "K50-train", 
		marker = :circle, markersize = 3, markerstrokewidth = 0.2, markerstrokecolor = :black,
		alpha = 0.8
	)

	scatter!(
		sorted_m50_posttrain_test, cum_prob_test, label = "M50-test",
		marker = :circle, markersize = 3, markerstrokewidth = 0.2, markerstrokecolor = :black, 
		alpha = 0.8
	)

	scatter!(
		sorted_k50_posttrain_test, cum_prob_test, label = "K50-test",
		marker = :circle, markersize = 3, markerstrokewidth = 0.2, markerstrokecolor = :black,
		alpha = 0.8
	)

	# Add median value lines with matching colors and annotations
	vline!([m50_train_median], label="M50-train median $(round(m50_train_median, digits=3))", linestyle=:dash, color=1, linewidth=2)

	vline!([k50_train_median], label="K50-train median $(round(k50_train_median, digits=3))", linestyle=:dash, color=2, linewidth=2) 

	vline!([m50_test_median], label="M50-test median $(round(m50_test_median, digits=3))", linestyle=:dash, color=3, linewidth=2)

	vline!([k50_test_median], label="K50-test median $(round(k50_test_median, digits=3))", linestyle=:dash, color=4, linewidth=2)

	xlabel!("$(criteria)", fontsize=12)
	ylabel!("Cumulative Probability", fontsize=12)
	plot!(framestyle = :box, grid = false)
	plot!(legend=:topleft)
	p
end

p = plot_cummulate_fig([m50_posttrain_result, k50_posttrain_result], "mnse")
savefig(p, "src/plots/mNSE_cummulate_fig.png")
