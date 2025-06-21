using HydroModels
using HydroModels: step_func, @variables, @parameters
# define variables and parameters
@variables temp lday pet prcp snowfall rainfall snowpack melt
@variables soilwater pet evap infil baseflow surfaceflow flow rainfall
@parameters Tmin Tmax Df Smax Qmax f

bucket = @hydrobucket :bucket begin
    fluxes = begin
        @hydroflux begin
            snowfall ~ step_func(Tmin - temp) * prcp
            rainfall ~ step_func(temp - Tmin) * prcp
        end
        @hydroflux melt ~ step_func(temp - Tmax) * step_func(snowpack) * minimum([snowpack, Df * (temp - Tmax)])
        @hydroflux infil ~ rainfall + melt
        @hydroflux pet ~ 29.8 * lday * 0.611 * exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)
        @hydroflux evap ~ step_func(soilwater) * step_func(soilwater - Smax) * pet +
                          step_func(soilwater) * step_func(Smax - soilwater) * pet * (soilwater / Smax)
        @hydroflux baseflow ~ step_func(soilwater) * step_func(soilwater - Smax) * Qmax +
                              step_func(soilwater) * step_func(Smax - soilwater) * Qmax * exp(-f * (Smax - soilwater))
        @hydroflux surfaceflow ~ step_func(soilwater) * step_func(soilwater - Smax) * (soilwater - Smax)
        @hydroflux flow ~ baseflow + surfaceflow
    end
    dfluxes = begin
        @stateflux snowpack ~ snowfall - melt
        @stateflux soilwater ~ rainfall + melt - (evap + flow)
    end
end

exphydro_model = @hydromodel :exphydro begin
    bucket
end