using ProgressMeter
using Optimization, OptimizationOptimisers, OptimizationBBO
using DataFrames, Dates
using Statistics

"""
	l1_loss(gamma) = (p) -> begin
	设定一个正则化系数, 对激活值进行正则化
"""
l1_loss(gamma) = (p) -> begin
    l1_temp = (abs.(p))
    activation_loss = sum(l1_temp)
    return activation_loss * gamma
end

"""
	reg_loss(gamma) = (p) -> begin
	设定一个正则化系数+熵正则化, 对激活值进行正则化
"""
reg_loss(gamma, params) = (p) -> begin
    l1_temp = abs.(Vector(p[params])) # [params]
    activation_loss = sum(l1_temp)
    entropy_temp = l1_temp ./ activation_loss
    entropy_loss = -sum(entropy_temp .* log.(entropy_temp))
    total_reg_loss = activation_loss + entropy_loss
    return total_reg_loss * gamma
end

mse_loss(obs, pred) = mean((pred .- obs) .^ 2)
nse_loss(obs, pred) = sum((pred .- obs) .^ 2) / sum((obs .- mean(obs)) .^ 2)
mnse_loss(obs, pred) = sum(abs.(pred .- obs)) / sum(abs.(obs .- mean(obs)))

function calibrate_exphydro(func, data, pas;
    lb, ub,
    loss_func=(y, y_hat) -> mean((y .- y_hat) .^ 2), warmup=100,
    optmzr=BBO_adaptive_de_rand_1_bin_radiuslimited(), max_N_iter=1000
)
    x, y, ts = data
    call_func(x, p) = func(x, p[1:6], p[7:8], ts) # for v2
    progress = Progress(max_N_iter, desc="Training...")
    recorder = []

    function objective(u, p)
        y_hat = call_func(x, u)
        return loss_func(y[warmup:length(y)], y_hat[warmup:length(y_hat)])
    end

    function callback(state, l)
        push!(recorder, (iter=state.iter, loss=l, time=now()))
        next!(progress)
        return false
    end

    optf = Optimization.OptimizationFunction(objective)
    optprob = Optimization.OptimizationProblem(optf, pas, lb=lb, ub=ub)
    sol = Optimization.solve(optprob, optmzr, maxiters=max_N_iter, callback=callback)
    recorder_df = DataFrame(recorder)
    return sol.u, recorder_df
end

"""
考虑initstates作为优化目标
"""
function calibrate_exphydro_pro1(func, data, pas;
    lb, ub,
    loss_func=(y, y_hat) -> mean((y .- y_hat) .^ 2), warmup=100,
    optmzr=BBO_adaptive_de_rand_1_bin_radiuslimited(), max_N_iter=1000
)
    x, y, ts = data
    call_func(x, p) = func(x, p[1:7], p[8:9], ts) # for v2
    # call_func(x, p) = func(x, p[1:6], p[7:8], ts) # for v1
    progress = Progress(max_N_iter, desc="Training...")
    recorder = []

    function objective(u, p)
        y_hat = call_func(x, u)
        return loss_func(y[warmup:length(y)], y_hat[warmup:length(y_hat)])
    end

    function callback(state, l)
        push!(recorder, (iter=state.iter, loss=l, time=now()))
        next!(progress)
        return false
    end

    optf = Optimization.OptimizationFunction(objective)
    optprob = Optimization.OptimizationProblem(optf, pas, lb=lb, ub=ub)
    sol = Optimization.solve(optprob, optmzr, maxiters=max_N_iter, callback=callback)
    recorder_df = DataFrame(recorder)
    return sol.u, recorder_df
end

"""
考虑initstates作为优化目标
"""
function calibrate_exphydro_pro2(func, data, pas;
    lb, ub,
    loss_func=(y, y_hat) -> mean((y .- y_hat) .^ 2), warmup=100,
    optmzr=BBO_adaptive_de_rand_1_bin_radiuslimited(), max_N_iter=1000
)
    x, y, ts = data
    call_func(x, p) = func(x, p[1:8], p[9:10], ts) # for v2
    # call_func(x, p) = func(x, p[1:6], p[7:8], ts) # for v1
    progress = Progress(max_N_iter, desc="Training...")
    recorder = []

    function objective(u, p)
        y_hat = call_func(x, u)
        return loss_func(y[warmup:length(y)], y_hat[warmup:length(y_hat)])
    end

    function callback(state, l)
        push!(recorder, (iter=state.iter, loss=l, time=now()))
        next!(progress)
        return false
    end

    optf = Optimization.OptimizationFunction(objective)
    optprob = Optimization.OptimizationProblem(optf, pas, lb=lb, ub=ub)
    sol = Optimization.solve(optprob, optmzr, maxiters=max_N_iter, callback=callback)
    recorder_df = DataFrame(recorder)
    return sol.u, recorder_df
end

function train_nn(nn_func, data, params;
    loss_func=(y, y_hat) -> mean((y .- y_hat) .^ 2), reg_func=(x) -> 0.0,
    optmzr=ADAM(0.01), max_N_iter=1000)
    x, y = data
    progress = Progress(max_N_iter, desc="Training...")
    recorder = []

    function objective(u, p)
        y_hat = nn_func(x, u)
        return loss_func(y, y_hat) + reg_func(u)
    end

    function callback(state, l)
        reg_loss = reg_func(state.u)
        push!(recorder, (
            iter=state.iter, loss=l - reg_loss,
            reg_loss=reg_loss,
            time=now()
        ))
        next!(progress)
        return false
    end

    optf = Optimization.OptimizationFunction(objective, Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, params)
    sol = Optimization.solve(optprob, optmzr, maxiters=max_N_iter, callback=callback)
    recorder_df = DataFrame(recorder)
    return sol.u, recorder_df
end

function train_nn_v2(nn_func, train_data, val_data, params;
    loss_func=(y, y_hat) -> mean((y .- y_hat) .^ 2), reg_func=(x) -> 0.0,
    optmzr=ADAM(0.01), max_N_iter=1000)
    x, y = train_data
    val_x, val_y = val_data
    progress = Progress(max_N_iter, desc="Training...")
    recorder = []

    function objective(u, p)
        y_hat = nn_func(x, u)
        return loss_func(y, y_hat) + reg_func(u)
    end

    function callback(state, l)
        reg_loss = reg_func(state.u)
        val_loss = loss_func(val_y, nn_func(val_x, state.u))
        push!(recorder, (
            iter=state.iter,
            train_loss=l - reg_loss,
            val_loss=val_loss,
            reg_loss=reg_loss,
            time=now()
        ))
        next!(progress)
        return false
    end

    optf = Optimization.OptimizationFunction(objective, Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, params)
    sol = Optimization.solve(optprob, optmzr, maxiters=max_N_iter, callback=callback)
    recorder_df = DataFrame(recorder)
    return sol.u, recorder_df
end

function train_hybrid(model_func, train_data, val_data, params;
    loss_func=(y, y_hat) -> mean((y .- y_hat) .^ 2), reg_func=(p) -> 0.0,
    optmzr=ADAM(0.01), max_N_iter=100, warm_up=100,
    adtype=Optimization.AutoZygote()
)
    train_x, train_y, train_timepoints = train_data
    total_x, total_y, total_timepoints = val_data
    train_len, total_len = length(train_y), length(total_y)
    progress = Progress(max_N_iter, desc="Training...")
    recorder = []
    ps_list = []

    function objective(u, _)
        train_y_hat = model_func(train_x, u)
        train_loss = loss_func(view(train_y, warm_up:train_len), view(train_y_hat, warm_up:train_len))
        return train_loss + reg_func(u)
    end

    function callback(state, l)
        total_y_hat = model_func(total_x, state.u)
        val_loss = loss_func(view(total_y, train_len:total_len), view(total_y_hat, train_len:total_len))
        push!(recorder, (iter=state.iter, train_loss=l - reg_func(state.u), val_loss=val_loss, reg_loss=reg_func(state.u), time=now()))
        push!(ps_list, state.u)
        next!(progress)
        return false
    end

    optf = Optimization.OptimizationFunction(objective, adtype)
    optprob = Optimization.OptimizationProblem(optf, params)
    sol = Optimization.solve(optprob, optmzr, maxiters=max_N_iter, callback=callback)
    recorder_df = DataFrame(recorder)
    # Get index of minimum validation loss
    best_idx = argmin(recorder_df.val_loss)

    return sol.u, ps_list[best_idx], recorder_df
end


function train_with_valv2(nn_func, train_data, val_data, params;
    loss_func=(y, y_hat) -> mean((y .- y_hat) .^ 2), reg_func=(p) -> 0.0,
    optmzr=ADAM(0.01), max_N_iter=100, warm_up=100, adtype=Optimization.AutoZygote()
)
    train_x, train_y, train_timepoints = train_data
    total_x, total_y, total_timepoints = val_data
    train_len, total_len = length(train_y), length(total_y)
    recorder = []
    ps_list = []

    function objective(u, p)
        train_y_hat = nn_func(train_x, u, train_timepoints)
        train_loss = loss_func(view(train_y, warm_up:train_len), view(train_y_hat, warm_up:train_len))
        return train_loss  # + reg_func(u)
    end

    function callback(state, l)
        total_y_hat = nn_func(total_x, state.u, total_timepoints)
        train_loss = loss_func(view(total_y, warm_up:train_len), view(total_y_hat, warm_up:train_len))
        val_loss = loss_func(view(total_y, train_len:total_len), view(total_y_hat, train_len:total_len))
        reg_loss = reg_func(state.u)
        @info "iter: $(state.iter), train_loss: $(train_loss), val_loss: $val_loss, reg_loss: $reg_loss, $(now())"
        push!(recorder, (iter=state.iter, train_loss=train_loss, val_loss=val_loss, reg_loss=reg_loss, time=now()))
        push!(ps_list, state.u)
        return false
    end

    optf = Optimization.OptimizationFunction(objective, adtype)
    optprob = Optimization.OptimizationProblem(optf, params)
    sol = Optimization.solve(optprob, optmzr, maxiters=max_N_iter, callback=callback)
    recorder_df = DataFrame(recorder)
    # Get index of minimum validation loss
    best_idx = argmin(recorder_df.val_loss)

    return sol.u, ps_list[best_idx], recorder_df
end


function train_symbolic_func(func, data, params;
    loss_func=(y, y_hat) -> mean((y .- y_hat) .^ 2), reg_func=(p) -> 0.0,
    optmzr=ADAM(0.001), max_N_iter=1000)
    x, y = data
    recorder = []

    function objective(u, p)
        y_hat = func.(eachslice(x, dims=2), Ref(u))
        return loss_func(y, y_hat)
    end

    function callback(state, l)
        reg_loss = reg_func(state.u)
        @info "iter: $(state.iter), loss: $(l - reg_loss), reg_loss: $reg_loss"
        push!(recorder, (iter=state.iter, loss=l - reg_loss, reg_loss=reg_loss, time=now()))
        # next!(progress)
        return false
    end

    optf = Optimization.OptimizationFunction(objective, Optimization.AutoForwardDiff())
    optprob = Optimization.OptimizationProblem(optf, params)
    sol = Optimization.solve(optprob, optmzr, maxiters=max_N_iter, callback=callback)
    recorder_df = DataFrame(recorder)
    return sol.u, recorder_df
end