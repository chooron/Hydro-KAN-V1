using CSV, DataFrames
using Symbolics

"""
提取由SymbolicRegression.jl生成的公式
"""
function extract_layer_eqs(formula_path)
	layer_folders = filter(x -> startswith(x, "layer_"), readdir(formula_path))
	layer_eqs = Dict()
	for layer_folder in layer_folders
		# Extract layer number and I/O mapping from folder name
		layer_match = match(r"layer_(\d+)\(I(\d+)-A(\d+)\)", layer_folder)
		layer_id = layer_match.captures[1]
		input_id = layer_match.captures[2]
		act_id = layer_match.captures[3]

		layer_path = joinpath(formula_path, layer_folder)

		# Get the most recent run folder
		run_folders = readdir(layer_path)
		latest_run = sort(run_folders)[end]

		# Read equations from hall_of_fame.csv
		hof_path = joinpath(layer_path, latest_run, "hall_of_fame.csv")
		hof_df = CSV.read(hof_path, DataFrame)

		# Store the full dataframe for this I->A mapping
		mapping_key = "layer$(layer_id)_I$(input_id)_A$(act_id)"
		layer_eqs[mapping_key] = hof_df
	end
	return layer_eqs
end

"""
提取表达式中的常数
"""
function find_numbers(ex)
    numbers = Number[]
    if ex isa Number
        push!(numbers, ex)
    elseif ex isa Expr
        for arg in ex.args
            append!(numbers, find_numbers(arg))
        end
    end
    return numbers
end

"""
将字符串的公式形式转换为expression
"""
function parse_to_func(expr_str; params_nm = :p)
	# Extract all floating point numbers from the string
	@variables x1
	parsed_expr = simplify(eval(Meta.parse(expr_str)))
	extracted_params = find_numbers(Symbolics.toexpr(parsed_expr))
	# Remove -1 from extracted parameters
	extracted_params = filter(x -> x != -1, extracted_params)
	@variables p[1:length(extracted_params)]
	subs_dict = Dict(param => p[i] for (i, param) in enumerate(extracted_params))
	for sub_dict in subs_dict
		parsed_expr = substitute.(parsed_expr, sub_dict)
	end
	@info round.(extracted_params, digits=4)
	@info parsed_expr
	tmp_func = build_function(parsed_expr, x1, p, expression = Val{false})
	# Print parameters with 4 decimal places
	return (x, p) -> tmp_func(x, p[params_nm]), NamedTuple{Tuple([params_nm])}([Float64.(extracted_params)])
end

"""
绘制拟合结果
"""
function check_fit_plot(tmp_func, params, input, target)
	# Sort input and get sorting indices
	sorted_indices = sortperm(input)
	sorted_target = target[sorted_indices]
	fig = plot(input[sorted_indices], tmp_func.(input, Ref(params))[sorted_indices])
	plot!(input[sorted_indices], sorted_target)
	return fig
end
