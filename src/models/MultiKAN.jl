using Random: AbstractRNG

using Lux
using KolmogorovArnold
using ComponentArrays
"""
    KAN{D,M} <: LuxCore.AbstractLuxLayer

Knowledge-Augmented Network (KAN) 是一个基于 Lux.jl 的神经网络层。

# 参数
- `D`: 层的类型
- `M`: 是否存在乘法操作的标志

# 字段
- `in_dims::Int`: 输入维度
- `out_dims::Int`: 输出维度
- `layers::NamedTuple`: 网络层的命名元组
- `mult_ids::Vector{Vector}`: 乘法操作的索引

# 示例
"""
struct KAN{D,M} <: LuxCore.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    layers::NamedTuple
    mult_ids::Vector{Vector}
    name::Symbol

    function KAN(width::Vector{Int}, mult_arity::Vector{<:Dict}, layer_cfg::Vector{<:Dict}, layer_type; name::Symbol)
        @assert length(width) == length(mult_arity) + 2 == length(layer_cfg) + 1 "input arguments must be of the same length"
        layers = layer_type[]
        in_dims, out_dims = width[1], width[end]
        new_width = copy(width)
        mult_ids = []
        for (i, wi) in enumerate(width[2:end-1])
            if !isempty(mult_arity[i])
                for k in 1:wi
                    if !(k in keys(mult_arity[i]))
                        mult_arity[i][k] = 1
                    end
                end
                new_width[i+1] = sum(values(mult_arity[i]))
                multi_arr = [mult_arity[i][k] for k in 1:wi]
                mult_id = [s:s+l-1 for (s, l) in zip(cumsum([1; multi_arr[1:end-1]]), multi_arr)]
                mult_ids = push!(mult_ids, mult_id)
            else
                mult_ids = push!(mult_ids, [])
            end
        end
        # for the last layer
        mult_ids = push!(mult_ids, [])
        exist_mult = all(x -> !isempty(x), mult_ids)
        for i in 1:length(new_width)-1
            grid_len_ = pop!(layer_cfg[i], :grid_len, 5)
            push!(layers, layer_type(width[i], new_width[i+1], grid_len_; layer_cfg[i]...))
        end
        layer_names = [Symbol("layer_$(i)") for i in 1:length(layers)]
        new{layer_type,exist_mult}(in_dims, out_dims, NamedTuple(zip(layer_names, layers)), mult_ids, name)
    end
end

@inline depth(k::KAN) = length(k.layers)

(k::KAN)(x, p, st) = applykan(k.layers, k.mult_ids, x, p, st)

function apply(layer, mult_id, x, p, st)
    y, st = layer(x, p, st)
    if isempty(mult_id)
        out = y
    else
        multi_x = reduce(hcat, map(mult_id) do id
            reduce(.*, eachslice(view(y, id, :), dims=1))
        end)
        out = permutedims(multi_x)
    end
    return out, st
end

@generated function applykan(layers::NamedTuple{fields}, mult_ids::Vector{Vector}, x, ps, st) where {fields}
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    st_symbols = [gensym() for _ in 1:N]
    calls = [:(($(x_symbols[i+1]), $(st_symbols[i])) = @inline apply(
        layers.$(fields[i]), mult_ids[$i], $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i])))
             for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(x_symbols[N+1]), st))
    return Expr(:block, calls...)
end

"""
    LuxCore.initialparameters(rng::AbstractRNG, k::KAN)
    initial parameters for KAN
"""
function LuxCore.initialparameters(rng::AbstractRNG, k::KAN)
    layer_names = [Symbol("layer_$(i)") for i in 1:depth(k)]
    return NamedTuple(zip(layer_names, [LuxCore.initialparameters(rng, k.layers[i]) for i in 1:depth(k)]))
end

function LuxCore.parameterlength(rng::AbstractRNG, k::KAN)
    return [LuxCore.parameterlength(rng, k.layers[i]) for i in 1:depth(k)] |> sum
end

function LuxCore.initialstates(rng::AbstractRNG, k::KAN)
    layer_names = [Symbol("layer_$(i)") for i in 1:depth(k)]
    return NamedTuple(zip(layer_names, [LuxCore.initialstates(rng, k.layers[i]) for i in 1:depth(k)]))
end

function LuxCore.statelength(rng::AbstractRNG, k::KAN)
    return [LuxCore.statelength(rng, k.layers[i]) for i in 1:depth(k)] |> sum
end

export KAN, depth