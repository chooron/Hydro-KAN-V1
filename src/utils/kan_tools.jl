# copy from ht,ps://github.com/DENG-MIT/K:N-ODEs
using JLD2
using Plots
using Lux
using ComponentArrays
using KolmogorovArnold
using StableRNGs
using CSV, DataFrames
using SymbolicRegression
using Statistics
using Printf
import SymbolicRegression: calculate_pareto_frontier

include("train.jl")

"""
	activation_getter(kan_layer, kan_params, kan_states, input_data)

	Get the activation of the KAN layer.
"""
function activation_getter(kan_layer, kan_params, kan_states, input_data)
    pc, pw = kan_params.C, kan_params.W
    # Extract pc list for each input dimension: (1:grid_len, grid_len+1:2*grid_len, 2*grid_len+1:3*grid_len)
    pc_list = [pc[:, ((i-1)*kan_layer.grid_len+1):(i*kan_layer.grid_len)] for i in 1:kan_layer.in_dims]
    # Extract pw for each input dimension 
    pw_list = [pw[:, i] for i in 1:kan_layer.in_dims]

    x = reshape(input_data, kan_layer.in_dims, :)
    K = size(x, 2)

    x_norm = kan_layer.normalizer(x)              # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, :)                        # [1, I * K]
    basis = kan_layer.basis_func(x_resh, kan_states.grid, kan_layer.denominator) # [G, I * K]
    # basis_resh = reshape(basis, kan_layer.grid_len, kan_layer.in_dims, K)
    # basis_resh_list = [basis_resh[:, i, :] for i in 1:kan_layer.in_dims]
    basis_list = [basis[:, i:kan_layer.in_dims:end] for i in 1:kan_layer.in_dims]
    splines_list = [basis_list[i]' * pc_list[i]' + kan_layer.base_act.(x[i, :]) .* pw_list[i]' for i in 1:kan_layer.in_dims]

    #sanity check: run the actual spline formulation and make sure they match
    basis_resh = reshape(basis, kan_layer.grid_len * kan_layer.in_dims, K)
    basis_resh2 = reshape(basis_resh, kan_layer.grid_len * kan_layer.in_dims, K)    # [G * I, K]
    spline_to_check = pc * basis_resh2 + pw * kan_layer.base_act.(x)
    @info sum(abs.(spline_to_check .- sum(splines_list)') .< 1e-10) == length(spline_to_check) #make sure it's all equal 

    return splines_list # G * [I, K]
end

"""
	prune(kan_layers, kan_params, postacts, theta = 1e-2)

	根据激活函数的scale进行剪枝并构建新的模型和参数
"""
function prune_qnn(kan_layers, kan_params, postacts, theta=1e-2)
    #pruning function used to sparsify KAN-ODEs (i.e. delete negligible connections)
    #theta corresponds to gamma_pr in the manuscript (value of 1e-2)
    #not optimized - only runs a few times per training cycle, so extreme efficiency is not important
    #make sure to turn regularization on (sparse_on=1 above) before pruning, otherwise it's unlikely anything will prune bc the KAN is not sparse
    layer_width = first(kan_layers).out_dims
    grid_size = first(kan_layers).grid_len
    postacts1, postacts2 = postacts

    max_postacts1 = vec(maximum(maximum(abs.(postacts1), dims=1)[1, :, :], dims=2))
    max_postacts2 = vec(maximum(abs.(postacts2), dims=1)[1, :, :])
    min_postact_score = vec(minimum([max_postacts1 max_postacts2], dims=2))
    sum_postact_score = sum(min_postact_score)
    @info "postact_score: $min_postact_score, threshold: $(sum_postact_score*theta) "
    nodes_to_keep = findall(x -> x > theta * sum_postact_score, min_postact_score)

    @info "kept nodes: $(length(nodes_to_keep))"

    ##re-initialize KAN, but with the smaller size
    pruned_layer_width = length(nodes_to_keep)

    new_kan_layers = [
        KDense(kan_layers[1].in_dims, pruned_layer_width, grid_size; use_base_act=true),
        KDense(pruned_layer_width, kan_layers[2].out_dims, grid_size; use_base_act=true),
    ]

    layer_1_params = kan_params.layer_1
    layer_2_params = kan_params.layer_2
    ##and save only those parameters into it
    pm1c = layer_1_params.C[nodes_to_keep, :]
    pm1w = layer_1_params.W[nodes_to_keep, :]
    pm2c = zeros(kan_layers[2].out_dims, grid_size * pruned_layer_width)
    count = 0
    for i in nodes_to_keep
        count += 1
        pm2c[:, (count-1)*grid_size+1:count*grid_size] = layer_2_params.C[:, (i-1)*grid_size+1:i*grid_size]
    end
    pm2w = layer_2_params.W[:, nodes_to_keep]

    pM_new = (layer_1=(C=pm1c, W=pm1w), layer_2=(C=pm2c, W=pm2w))
    return new_kan_layers, pM_new, nodes_to_keep, min_postact_score
end

function prune_etnn(kan_layers, kan_params, postacts, theta=1e-2)
    #pruning function used to sparsify KAN-ODEs (i.e. delete negligible connections)
    #theta corresponds to gamma_pr in the manuscript (value of 1e-2)
    #not optimized - only runs a few times per training cycle, so extreme efficiency is not important
    #make sure to turn regularization on (sparse_on=1 above) before pruning, otherwise it's unlikely anything will prune bc the KAN is not sparse
    
    # layer_width = first(kan_layers).out_dims # This might be different if it's the final layer
    grid_size = first(kan_layers).grid_len
    postacts1 = first(postacts) # Expecting a single array for postacts

    max_postacts1 = vec(maximum(maximum(abs.(postacts1), dims=1)[1, :, :], dims=2))
    # Removed max_postacts2 calculation
    min_postact_score = max_postacts1 # Simplified as there's only one postact
    sum_postact_score = sum(min_postact_score)
    @info "postact_score: $min_postact_score, threshold: $(sum_postact_score*theta) "
    nodes_to_keep = findall(x -> x > theta * sum_postact_score, min_postact_score)

    @info "kept nodes: $(length(nodes_to_keep))"

    ##re-initialize KAN, but with the smaller size
    pruned_layer_width = length(nodes_to_keep)

    new_kan_layers = [
        KDense(kan_layers[1].in_dims, pruned_layer_width, grid_size; use_base_act=true),
        kan_layers[2]
    ]

    layer_1_params = kan_params.layer_1
    layer_2_params = kan_params.layer_2
    # Removed layer_2_params
    ##and save only those parameters into it
    pm1c = layer_1_params.C[nodes_to_keep, :]
    pm1w = layer_1_params.W[nodes_to_keep, :]
    # Removed pm2c and pm2w calculations

    pM_new = (layer_1=(C=pm1c, W=pm1w),layer_2=layer_2_params) # Removed layer_2 parameters
    return new_kan_layers, pM_new, nodes_to_keep, min_postact_score
end

"""
	obtain_nn_splines(;	model_layers,input_data,layer_params)
"""
function obtain_qnn_splines(; model_layers, input_data, layer_params)
    layer_1, layer_2 = model_layers[1], model_layers[2]
    layer_1_states = LuxCore.initialstates(StableRNG(1234), layer_1)
    layer_2_states = LuxCore.initialstates(StableRNG(1234), layer_2)

    layer1_postacts = activation_getter(layer_1, layer_params.layer_1, layer_1_states, input_data)
    layer1_output = layer_1(input_data, layer_params.layer_1, layer_1_states)[1]
    layer2_postacts = activation_getter(layer_2, layer_params.layer_2, layer_2_states, layer1_output)

    layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), layer1_postacts) # K * O * I (K * 6 * 3)
    layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), layer2_postacts) # K * O * I (K * 1 * 6)

    splines_dict = [
        "postacts1" => layer1_postacts_arr, "postacts2" => layer2_postacts_arr,
        "acts1" => input_data, "acts2" => layer1_output,
    ]
    Dict(splines_dict...)
end

function obtain_etnn_splines(; model_layers, input_data, layer_params)
    layer_1 = model_layers[1]
    layer_1_states = LuxCore.initialstates(StableRNG(1234), layer_1)

    layer1_postacts = activation_getter(layer_1, layer_params.layer_1, layer_1_states, input_data)
    layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), layer1_postacts) # K * O * I (K * 6 * 3)

    splines_dict = [
        "postacts1" => layer1_postacts_arr,
        "acts1" => input_data,
    ]
    Dict(splines_dict...)
end

"""
	prune_edge(splines; threshold = 1e-1)

	根据spline的上下限, 计算出需要剪枝的边

	splines: K * I * A (time series, input dims, activation funcs)
	threshold: 阈值, 默认1e-1
"""
function prune_edge(splines; threshold=1e-1)
    # splines dims: K * I * A (time series, input dims, activation funcs)
    spline_scale = maximum(splines, dims=1)[1, :, :] - minimum(splines, dims=1)[1, :, :]
    # Normalize by dividing by maximum value
    norm_stats = spline_scale ./ maximum(spline_scale)

    # Get indices where normalized value is below threshold
    prune_indices = findall(x -> x < threshold, norm_stats)

    return prune_indices # input dims => activation funcs will be pruned
end

"""
	prune_nodes(; model_layers, layer_params, input_data, prune_threshold)
	根据输入数据和模型参数, 剪枝隐藏层节点
"""
function prune_qnn_nodes(; model_layers, layer_params, input_data, prune_threshold)
    spline_dict = obtain_qnn_splines(model_layers=model_layers, layer_params=layer_params, input_data=input_data)
    postacts1, postacts2 = spline_dict["postacts1"], spline_dict["postacts2"]
    new_kan_layers, new_ps, nodes_to_keep, node_scores = prune_qnn(model_layers, layer_params, [postacts1, postacts2], prune_threshold)
    return new_kan_layers, new_ps, nodes_to_keep, node_scores
end

function prune_etnn_nodes(; model_layers, layer_params, input_data, prune_threshold)
    spline_dict = obtain_etnn_splines(model_layers=model_layers, layer_params=layer_params, input_data=input_data)
    postacts1 = spline_dict["postacts1"]
    new_kan_layers, new_ps, nodes_to_keep, node_scores = prune_etnn(model_layers, layer_params, [postacts1], prune_threshold)
    return new_kan_layers, new_ps, nodes_to_keep, node_scores
end

"""
	绘制激活函数并根据scale显示不同的透明度
"""
function plot_activation_functions(acts, postacts; alpha_scale=1.0)
    input_dims = size(acts, 1)
    activ_dims = size(postacts, 2)

    # 创建一个网格布局的图
    p = plot(layout=(activ_dims, input_dims),
        size=(200 * input_dims, 200 * activ_dims),
        legend=false)

    # 对每个输入维度进行循环
    for i in axes(acts, 1)
        current_acts = acts[i, :]
        # Sort acts and get indices
        sorted_indices = sortperm(current_acts)
        acts_scale = maximum(current_acts) - minimum(current_acts)

        # 对每个激活函数进行循环
        for j in axes(postacts, 2)
            current_postacts = postacts[:, j, i]

            # 计算透明度
            alpha = tanh(alpha_scale * (maximum(current_postacts) - minimum(current_postacts)) / acts_scale)
            color = alpha < 1e-1 ? :grey : :blue
            alpha = alpha < 1e-1 ? 0.5 : alpha

            # 计算当前子图的索引
            plot_idx = (j - 1) * input_dims + i

            x_range = extrema(current_acts)
            y_range = extrema(current_postacts)
            xticks = range(x_range[1], x_range[2], length=3)
            yticks = range(y_range[1], y_range[2], length=3)

            # 在对应的子图中绘制线条
            plot!(p[plot_idx], current_acts[sorted_indices], current_postacts[sorted_indices],
                alpha=alpha,
                linewidth=2,
                color=color,
                xticks=xticks,
                yticks=yticks,
                tickfontsize=8,
                xformatter=x -> @sprintf("%.2f", x),  # 格式化为2位小数
                yformatter=y -> @sprintf("%.2f", y))

            # 在图内添加标注
            annotate!(p[plot_idx], x_range[1] + 0.1 * (x_range[2] - x_range[1]),
                y_range[2] - 0.1 * (y_range[2] - y_range[1]), text("I$i→A$j", 10, :left))
        end
    end

    # 调整整体布局
    plot!(p, margin=5Plots.mm)

    return p
end

"""
	stratified_sample(x, y, n_bins=10, samples_per_bin=100)

Takes input x and corresponding y values and returns stratified samples by dividing x into bins
and sampling equally from each bin to ensure better representation across the full range.

Arguments:
- x: Input array to sample from
- y: Corresponding output array 
- n_bins: Number of bins to divide x range into
- samples_per_bin: Number of samples to take from each bin

Returns:
- Tuple of sampled (x, y) arrays
"""
function stratified_sample(x, y; n_bins=5, samples_per_bin=100)
    # Ensure x and y are vectors
    x = vec(x)
    y = vec(y)

    # Calculate bin edges
    edges = range(minimum(x), maximum(x), length=n_bins + 1)

    # Initialize arrays to store samples
    sampled_x = Float64[]
    sampled_y = Float64[]

    # Sample from each bin
    for i in 1:n_bins
        # Find points in current bin
        bin_mask = (x .>= edges[i]) .& (x .< edges[i+1])
        bin_indices = findall(bin_mask)

        if !isempty(bin_indices)
            # If we have fewer points than requested samples, take all points
            n_samples = min(samples_per_bin, length(bin_indices))

            # Randomly sample indices from this bin
            sampled_indices = bin_indices[rand(1:length(bin_indices), n_samples)]
            println((length(bin_indices), length(sampled_indices)))
            @info "sampled_indices: $(length(bin_indices)), $(length(sampled_indices))"
            # Add sampled points to our arrays
            append!(sampled_x, x[sampled_indices])
            append!(sampled_y, y[sampled_indices])
        end
    end

    # Return as tuple of arrays
    return reshape(sampled_x, 1, :), reshape(sampled_y, 1, :)
end

"""
	fit_kan_activations(kan_model, input_data, layers_ps, model_name)

	根据输入数据和KAN模型, 计算出每个激活函数的表达式, 默认是2层的模型
"""
function fit_kan_activations(kan_model, input_data, layers_ps, save_dir)
    layer_1_states = LuxCore.initialstates(StableRNG(1234), kan_model.layer_1)
    layer_2_states = LuxCore.initialstates(StableRNG(1234), kan_model.layer_2)

    layer1_postacts = activation_getter(kan_model.layer_1, layers_ps.layer_1, layer_1_states, input_data)
    layer1_output = kan_model.layer_1(input_data, layers_ps.layer_1, layer_1_states)[1]
    layer2_postacts = activation_getter(kan_model.layer_2, layers_ps.layer_2, layer_2_states, layer1_output)

    layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), layer1_postacts) # K * O * I (K * 6 * 3)
    layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), layer2_postacts) # K * O * I (K * 1 * 6)

    layer1_dominating_eqs = Dict()
    # fit the first layer
    for i in axes(layer1_postacts_arr, 3)
        tmp_x = input_data[i:i, :]
        for j in axes(layer1_postacts_arr, 2)
            options = Options(
                binary_operators=[+, *, -],
                unary_operators=[exp],
                populations=20,
                maxsize=15,
                expression_options=(; max_parameters=5),
                elementwise_loss=L2DistLoss(),
                output_directory=save_dir * "/layer_1(I$i-A$j)",
            )
            @info "fitting layer 1 eq I$i=>A$j"
            tmp_y = permutedims(layer1_postacts_arr[:, j, i:i])
            @info "tmp_x: $(size(tmp_x)), tmp_y: $(size(tmp_y))"
            sampled_x, sampled_y = stratified_sample(tmp_x, tmp_y)
            @info "sampled_x: $(size(sampled_x)), sampled_y: $(size(sampled_y))"
            hall_of_fame = equation_search(
                sampled_x, sampled_y, niterations=100, options=options,
                parallelism=:multithreading,
            )
            dominating = calculate_pareto_frontier(hall_of_fame)
            layer1_dominating_eqs["I$i=>A$j"] = dominating
        end
    end

    layer2_dominating_eqs = Dict()
    # fit the second layer
    for i in axes(layer2_postacts_arr, 3)
        tmp_x = layer1_output[i:i, :]
        for j in axes(layer2_postacts_arr, 2)
            options = Options(
                binary_operators=[+, *, -], # ^ 不支持负值所以不采用
                unary_operators=[exp],
                populations=20,
                maxsize=15,
                expression_options=(; max_parameters=5),
                elementwise_loss=L2DistLoss(),
                output_directory=save_dir * "/layer_2(I$i-A$j)",
            )
            @info "fitting layer 2 eq I$i=>A$j"
            tmp_y = permutedims(layer2_postacts_arr[:, j, i:i])
            println(size(tmp_x), size(tmp_y))
            sampled_x, sampled_y = stratified_sample(tmp_x, tmp_y)
            println(size(sampled_x), size(sampled_y))
            hall_of_fame = equation_search(
                sampled_x, sampled_y, niterations=100, options=options,
                parallelism=:multithreading,
            )
            dominating = calculate_pareto_frontier(hall_of_fame)
            layer2_dominating_eqs["I$i=>A$j"] = dominating
        end
    end
end
