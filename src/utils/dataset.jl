using CSV
using DataFrames
using JLD2
using MLUtils

function load_m50_est_dataset(basin_id::String; seq_len::Int = 20, batchsize::Int = 128)
	# Load exphydro data
	exphydro_df = CSV.read("data/exphydro/$(basin_id).csv", DataFrame)
	snowpack = exphydro_df[!, "snowpack"]
	soilwater = exphydro_df[!, "soilwater"]

	s0_max, s0_min = maximum(snowpack), minimum(snowpack)
	s1_max, s1_min = maximum(soilwater), minimum(soilwater)
	snowpack_norm = (snowpack .- s0_min) ./ (s0_max - s0_min)
	soilwater_norm = (soilwater .- s1_min) ./ (s1_max - s1_min)

	# Load CAMELS data
	camels_data = load("data/camelsus/$(basin_id).jld2")
	data_x = camels_data["data_x"]

	# Extract features from CAMELS data
	n_samples = size(data_x, 1) - seq_len + 1
	n_features = size(data_x, 2)

	# Initialize arrays
	X = zeros(n_samples, seq_len, n_features)
	y = zeros(n_samples, 2)

	# Create sliding window samples
	for i in 1:n_samples
		X[i, :, :] = data_x[i:i+seq_len-1, :]
		y[i, :] = [snowpack_norm[i+seq_len-1], soilwater_norm[i+seq_len-1]]
	end

	X_perm = permutedims(X, (2, 3, 1))
	y_perm = permutedims(y, (2, 1))
	(x_train, y_train), (x_val, y_val) = splitobs((X_perm, y_perm); at = 0.8, shuffle = true)

	# Use DataLoader to automatically minibatch and shuffle the data
	x_dataloader = DataLoader(collect.((permutedims(x_train, (2, 1, 3)), y_train)); batchsize = batchsize, shuffle = true)
	# Don't shuffle the validation data
	y_dataloader = DataLoader(collect.((permutedims(x_val, (2, 1, 3)), y_val)); batchsize = batchsize, shuffle = false)
	return x_dataloader, y_dataloader
end


function load_m50_m_dataset_states(basin_id::String; initstates = [], sub_len::Int = 100)
	# Load CAMELS data
	camelsus_cache = load("data/camelsus/$(basin_id).jld2")
	data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
	train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
	test_x, test_y, test_timepoints = camelsus_cache["test_x"], camelsus_cache["test_y"], camelsus_cache["test_timepoints"]
	# load exphydro data
	exphydro_df = CSV.read("data/exphydro/$(basin_id).csv", DataFrame)
	snowpack = exphydro_df[!, "snowpack"]
	soilwater = exphydro_df[!, "soilwater"]

	# Calculate dimensions
	n_features = size(data_x, 2)
	n_train_samples = div(size(train_x, 1), sub_len)
	n_test_samples = div(size(test_x, 1), sub_len)

	# Initialize arrays
	train_x_arr = zeros(n_features, sub_len, n_train_samples) # (var_dim, sub_len, n_samples)
	train_y_arr = zeros(sub_len, n_train_samples) # (sub_len, n_samples) 
	train_st_arr = zeros(2, n_train_samples) # (var_dim, n_samples)
	train_ts_arr = zeros(sub_len, n_train_samples) # (var_dim, n_samples)

	# Fill arrays using sliding windows
	for i in 1:n_train_samples
		sample_start = (i - 1) * sub_len + 1
		sample_end = i * sub_len

		# Fill x with current window
		train_x_arr[:, :, i] = permutedims(data_x[sample_start:sample_end, :])

		# Fill y with corresponding target values
		train_y_arr[:, i] = data_y[sample_start:sample_end]

		# Fill s with historical states before current window
		train_st_arr[:, i] = sample_start - 1 == 0 ? initstates : [snowpack[sample_start-1], soilwater[sample_start-1]]

        train_ts_arr[:, i] = data_timepoints[sample_start:sample_end]
	end

	test_x_arr = zeros(n_features, sub_len, n_test_samples) # (var_dim, sub_len, n_samples)
	test_y_arr = zeros(sub_len, n_test_samples) # (sub_len, n_samples) 
	test_st_arr = zeros(2, n_test_samples) # (var_dim, n_samples)
	test_ts_arr = zeros(sub_len, n_test_samples) # (var_dim, n_samples)

	test_idx_start = size(train_x, 1) + 1
	for i in 1:n_test_samples
		sample_start = test_idx_start + (i - 1) * sub_len
		sample_end = sample_start + sub_len - 1
		test_x_arr[:, :, i] = permutedims(data_x[sample_start:sample_end, :])
		test_y_arr[:, i] = data_y[sample_start:sample_end]
		test_st_arr[:, i] = [snowpack[sample_start-1], soilwater[sample_start-1]]
        test_ts_arr[:, i] = data_timepoints[sample_start:sample_end]
	end
	train_dataloader = DataLoader(collect.((train_x_arr, train_st_arr, train_ts_arr, train_y_arr)); batchsize = n_train_samples, shuffle = true)
	test_dataloader = DataLoader(collect.((test_x_arr, test_st_arr, test_ts_arr, test_y_arr)); batchsize = n_test_samples, shuffle = false)
	return train_dataloader, test_dataloader
end


function load_m50_m_dataset_hist(basin_id::String; sub_len::Int = 100, seq_len::Int = 20)
	# Load CAMELS data
	camelsus_cache = load("data/camelsus/$(basin_id).jld2")
	data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
	train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
	test_x, test_y, test_timepoints = camelsus_cache["test_x"], camelsus_cache["test_y"], camelsus_cache["test_timepoints"]
	# Calculate dimensions
	n_features = size(data_x, 2)
	n_train_samples = div(size(train_x, 1) - seq_len, sub_len)
	n_test_samples = div(size(test_x, 1), sub_len)

	# Initialize arrays
	train_x_arr = zeros(n_features, sub_len, n_train_samples) # (var_dim, sub_len, n_samples)
	train_y_arr = zeros(sub_len, n_train_samples) # (sub_len, n_samples) 
	train_h_arr = zeros(n_features, seq_len, n_train_samples) # (var_dim, seq_len, n_samples)

	test_x_arr = zeros(n_features, sub_len, n_test_samples) # (var_dim, sub_len, n_samples)
	test_y_arr = zeros(sub_len, n_test_samples) # (sub_len, n_samples) 
	test_h_arr = zeros(n_features, seq_len, n_test_samples) # (var_dim, seq_len, n_samples)

	# Fill arrays using sliding windows
	for i in 1:n_train_samples
		sample_start = (i - 1) * sub_len + seq_len + 1
		sample_end = i * sub_len + seq_len

		# Fill x with current window
		train_x_arr[:, :, i] = permutedims(data_x[sample_start:sample_end, :])

		# Fill y with corresponding target values
		train_y_arr[:, i] = data_y[sample_start:sample_end]

		# Fill h with historical data before current window
		train_h_arr[:, :, i] = permutedims(data_x[sample_start-seq_len:sample_start-1, :])
	end

	test_idx_start = size(train_x, 1) + 1
	for i in 1:n_test_samples
		sample_start = test_idx_start + (i - 1) * sub_len
		sample_end = sample_start + sub_len - 1
		test_x_arr[:, :, i] = permutedims(data_x[sample_start:sample_end, :])
		test_y_arr[:, i] = data_y[sample_start:sample_end]
		test_h_arr[:, :, i] = permutedims(data_x[sample_start-seq_len:sample_start-1, :])
	end
	train_dataloader = DataLoader(collect.((train_x_arr, train_y_arr, train_h_arr)); batchsize = 128, shuffle = true)
	test_dataloader = DataLoader(collect.((test_x_arr, test_y_arr, test_h_arr)); batchsize = 128, shuffle = false)
	return train_dataloader, test_dataloader
end
