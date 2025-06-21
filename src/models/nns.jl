using Lux
using KolmogorovArnold
include("MultiKAN.jl")

function build_mlp_nns(hidd_dims=16)
    etnn = Lux.Chain(
        Lux.Dense(3, hidd_dims, tanh),
        Lux.Dense(hidd_dims, hidd_dims, leakyrelu),
        Lux.Dense(hidd_dims, 1, leakyrelu),
        name=:etnn,
    )
    #* input is norm_soilwater, norm_prcp
    qnn = Lux.Chain(
        Lux.Dense(3, hidd_dims, tanh),
        Lux.Dense(hidd_dims, hidd_dims, leakyrelu),
        Lux.Dense(hidd_dims, 1, leakyrelu),
        name=:qnn,
    )
    return etnn, qnn
end

function build_kanv1_nns(hidd_dims=4, grid_size=4, basis_func=rbf)
    #* input is norm_snowpack, norm_soilwater, norm_temp, output is evap
    normalizer = tanh_fast
    etnn = Chain(
        KDense(3, hidd_dims, grid_size; use_base_act=true, basis_func, normalizer),
        KDense(hidd_dims, 1, grid_size; use_base_act=true, basis_func, normalizer),
        name=:etnn,
    )
    #* input is norm_soilwater, norm_prcp
    qnn = Chain(
        KDense(3, hidd_dims, grid_size; use_base_act=true, basis_func, normalizer),
        KDense(hidd_dims, 1, grid_size; use_base_act=true, basis_func, normalizer),
        name=:qnn,
    )
    return etnn, qnn
end

function build_kanv2_nns(hidd_dims=4, grid_size=4, basis_func=rbf)
    #* input is norm_snowpack, norm_soilwater, norm_temp, output is evap
    layer_config = Dict(:grid_len => grid_size, :normalizer => tanh_fast, :basis_func => basis_func)
    # split nn
    etnn = KAN([3, hidd_dims, 1], [Dict(1 => 2, 2 => 2)], [layer_config, layer_config], KDense; name=:etnn)
    # recharge nn
    qnn = KAN([3, hidd_dims, 1], [Dict(1 => 2, 2 => 2)], [layer_config, layer_config], KDense; name=:qnn)
    return etnn, qnn
end

function build_kanv3_nns(hidd_dims=4, grid_size=4, basis_func=rbf)
    #* input is norm_snowpack, norm_soilwater, norm_temp, output is evap
    normalizer = tanh_fast
    etnn = Chain(
        KDense(4, 3, 6; use_base_act=true, basis_func, normalizer),
        (x) -> prod(x, dims=1),
        name=:etnn,
    )
    #* input is norm_soilwater, norm_prcp
    qnn = Chain(
        KDense(3, hidd_dims, grid_size; use_base_act=true, basis_func, normalizer),
        KDense(hidd_dims, 1, grid_size; use_base_act=true, basis_func, normalizer),
        name=:qnn,
    )
    return etnn, qnn
end

function build_graphkan_nns(hidd_dims=4, grid_size=4, basis_func=rbf)
    etnn = Chain(
        Dense(3, hidd_dims, tanh), # equal to nn.Linear(3, hidd_dims)
        KDense(hidd_dims, hidd_dims, grid_size; use_base_act=true, basis_func, tanh_fast),
        Dense(hidd_dims, 1, leakyrelu), # equal to nn.Linear(hidd_dims, 1)
        name=:etnn,
    )
    qnn = Chain(
        Dense(3, hidd_dims, tanh), # equal to nn.Linear(3, hidd_dims)
        KDense(hidd_dims, hidd_dims, grid_size; use_base_act=true, basis_func, tanh_fast),
        Dense(hidd_dims, 1, leakyrelu), # equal to nn.Linear(hidd_dims, 1)
        name=:qnn,
    )
    return etnn, qnn
end
