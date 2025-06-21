using Statistics

function rmse(y, y_hat)
    return sqrt(mean((y .- y_hat) .^ 2))
end

function mae(y, y_hat)
    return mean(abs.(y .- y_hat))
end

function nse(y, y_hat)
    return 1 - sum((y .- y_hat) .^ 2) / sum((y .- mean(y)) .^ 2)
end

function mnse(y, y_hat)
    return 1 - sum(abs.(y .- y_hat)) / sum(abs.(y .- mean(y)))
end

function fhv(y, y_hat; h=0.02)
    # get arrays of sorted (descending) discharges
    obs = sort(y, rev=true)
    sim = sort(y_hat, rev=true)
    
    # subset data to only top h flow values
    n = round(Int, h * length(obs))
    obs_top = obs[1:n]
    sim_top = sim[1:n]
    
    # calculate fhv bias
    fhv = sum(abs.(sim_top - obs_top)) / sum(abs.(obs_top))
    
    return fhv * 100
end

function kge(y, y_hat; weights=[1.0, 1.0, 1.0])
    # Calculate Pearson correlation coefficient
    r = cor(y, y_hat)
    
    # Calculate alpha (ratio of standard deviations)
    alpha = std(y_hat) / std(y)
    
    # Calculate beta (ratio of means)
    beta = mean(y_hat) / mean(y)
    
    # Calculate KGE
    value = weights[1] * (r - 1)^2 + weights[2] * (alpha - 1)^2 + weights[3] * (beta - 1)^2
    
    return 1 - sqrt(value)
end

