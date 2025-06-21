using HydroModels
using Lux
using KolmogorovArnold

step_func(x) = (tanh(5.0 * x) + 1.0) * 0.5

function build_nns()
    basis_func = rbf
    normalizer = tanh_fast
    etnn = Chain(
        KDense(4, 3, 6; use_base_act=true, basis_func, normalizer),
        (x) -> prod(x, dims=1),
        name=:etnn,
    )
    qnn = Chain(
        KDense(3, 6, 6; use_base_act=true, basis_func, normalizer),
        KDense(6, 1, 6; use_base_act=true, basis_func, normalizer),
        name=:qnn,
    )
    return etnn, qnn
end

function build_m50_model(ep_nn, q_nn, norm_params)
    # Model parameters
    # Physical process parameters
    @parameters Tmin Tmax Df Smax f Qmax

    snowpack_std, snowpack_mean = norm_params[1], norm_params[2]
    soilwater_std, soilwater_mean = norm_params[3], norm_params[4]
    temp_std, temp_mean = norm_params[5], norm_params[6]
    lday_std, lday_mean = norm_params[7], norm_params[8]
    infil_std, infil_mean = norm_params[9], norm_params[10]

    # Model variables
    # Input variables
    @variables prcp temp lday
    # State variables
    @variables snowpack soilwater
    # Process variables
    @variables rainfall snowfall melt infil
    # Neural network variables
    @variables evap baseflow surfaceflow flow
    @variables norm_snw norm_slw norm_temp norm_infil norm_lday

    # Soil water component
    bucket = @hydrobucket :bucket begin
        fluxes = begin
            @hydroflux begin
                snowfall ~ step_func(Tmin - temp) * prcp
                rainfall ~ step_func(temp - Tmin) * prcp
            end
            @hydroflux melt ~ step_func(temp - Tmax) * min(snowpack, Df * (temp - Tmax))
            @hydroflux infil ~ rainfall + melt

            @hydroflux norm_snw ~ (snowpack - snowpack_mean) / snowpack_std
            @hydroflux norm_slw ~ (soilwater - soilwater_mean) / soilwater_std
            @hydroflux norm_infil ~ (infil - infil_mean) / infil_std
            @hydroflux norm_lday ~ (lday - lday_mean) / lday_std
            @hydroflux norm_temp ~ (temp - temp_mean) / temp_std
            @neuralflux evap ~ ep_nn([norm_snw, norm_slw, norm_temp, norm_lday])
            @neuralflux flow ~ q_nn([norm_snw, norm_slw, norm_infil])
        end
        dfluxes = begin
            @stateflux snowpack ~ snowfall - melt
            @stateflux soilwater ~ rainfall + melt - evap - flow
        end
    end

    # Complete model
    m50_model = @hydromodel :m50 begin
        bucket
    end
    return m50_model
end

function build_m25_model(q_nn, norm_params)
    # Model parameters
    # Physical process parameters
    @parameters Tmin Tmax Df Smax f Qmax

    snowpack_std, snowpack_mean = norm_params[1], norm_params[2]
    soilwater_std, soilwater_mean = norm_params[3], norm_params[4]
    infil_std, infil_mean = norm_params[5], norm_params[6]

    # Model variables
    # Input variables
    @variables prcp temp lday
    # State variables
    @variables snowpack soilwater
    # Process variables
    @variables rainfall snowfall melt infil pet
    # Neural network variables
    @variables evap baseflow surfaceflow flow
    @variables norm_snw norm_slw norm_infil

    # Soil water component
    bucket = @hydrobucket :bucket begin
        fluxes = begin
            @hydroflux begin
                snowfall ~ step_func(Tmin - temp) * prcp
                rainfall ~ step_func(temp - Tmin) * prcp
            end
            @hydroflux melt ~ step_func(temp - Tmax) * min(snowpack, Df * (temp - Tmax))
            @hydroflux infil ~ rainfall + melt

            @hydroflux norm_snw ~ (snowpack - snowpack_mean) / snowpack_std
            @hydroflux norm_slw ~ (soilwater - soilwater_mean) / soilwater_std
            @hydroflux norm_infil ~ (infil - infil_mean) / infil_std
            @hydroflux pet ~ 29.8 * lday * 0.611 * exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)
            @hydroflux evap ~ step_func(soilwater) * step_func(soilwater - Smax) * pet +
                              step_func(soilwater) * step_func(Smax - soilwater) * pet * (soilwater / Smax)
            @neuralflux flow ~ q_nn([norm_snw, norm_slw, norm_infil])
        end
        dfluxes = begin
            @stateflux snowpack ~ snowfall - melt
            @stateflux soilwater ~ rainfall + melt - evap - flow
        end
    end

    # Complete model
    m50_model = @hydromodel :m25 begin
        bucket
    end
    return m50_model
end
