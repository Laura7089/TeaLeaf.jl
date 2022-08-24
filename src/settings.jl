export Settings
export CONDUCTIVITY, RECIP_CONDUCTIVITY
export checkingvalue
export resettoexchange!
export debugrecord

# The accepted types of state geometry
@enum Geometry Rectangular Circular Point

const CONDUCTIVITY = 1
const RECIP_CONDUCTIVITY = 2

# TODO: can we make this immutable?
@with_kw mutable struct State
    density::Float64 = 0.0
    energy::Float64 = 0.0
    xmin::Float64 = 0.0
    ymin::Float64 = 0.0
    xmax::Float64 = 0.0
    ymax::Float64 = 0.0
    radius::Float64 = 0.0
    geometry::Geometry = Rectangular
end

# The main settings structure
@with_kw mutable struct Settings
    # Solve-wide constants
    endstep::Int = typemax(Int)
    presteps::Int = 30
    maxiters::Int = 10_000
    coefficient::Int = CONDUCTIVITY
    ppcginnersteps::Int = 10
    summaryfrequency::Int = 10
    halodepth::Int = 2

    # Fields to exchange
    toexchange::Dict{Symbol,Bool} =
        Dict(zip(CHUNK_EXCHANGE_FIELDS, fill(false, length(CHUNK_EXCHANGE_FIELDS))))

    errorswitch::Bool = false
    checkresult::Bool = true

    eps::Float64 = 1e-15
    dtinit::Float64 = 0.1
    endtime::Float64 = 10.0
    epslim::Float64 = 1e-5

    solver::Module = CG

    # Field dimensions
    xcells::Int = 10
    ycells::Int = 10

    xmin::Float64 = 0.0
    ymin::Float64 = 0.0
    xmax::Float64 = 100.0
    ymax::Float64 = 100.0

    dx::Float64 = (xmax - xmin) / xcells
    dy::Float64 = (ymax - ymin) / ycells

    states::Vector{State} = []

    debugfile::String = ""
end
Broadcast.broadcastable(s::Settings) = Ref(s)
resettoexchange!(s::Settings) = setindex!.(Ref(s.toexchange), false, CHUNK_EXCHANGE_FIELDS)

function Settings(infile)
    settings = Settings()
    # Open the configuration file
    @info "Reading configuration from $(infile)"
    open(infile, read = true) do tea_in
        while !eof(tea_in)
            line = readline(tea_in)

            if startswith(line, "state")
                push!(settings.states, readstate(line, settings))
                continue
            end

            # Skip comments and misunderstood lines
            if startswith(line, "*") || !occursin("=", line)
                continue
            end

            # TODO: respect the `use_[solver]` lines in the config

            # Read all of the settings from the config
            key, val = split(line, "=")
            key = Symbol(strip(replace(key, "_" => "")))
            key = key == :initialtimestep ? :dtinit : key
            try
                val = parse(fieldtype(Settings, key), val)
                setfield!(settings, key, val)
            catch
                @warn "Unknown setting" line key
            end
        end
    end

    settings.dx = (settings.xmax - settings.xmin) / settings.xcells
    settings.dy = (settings.ymax - settings.ymin) / settings.ycells
    return settings
end

function readstate(line, settings::Settings)::State
    thesplit = split(line, " ")

    state_num = parse(Int, thesplit[2])
    state = State()

    for pair in thesplit[3:end]
        (key, val) = split(pair, "=")

        @match key begin
            "density" => (state.density = parse(Float64, val))
            "energy" => (state.energy = parse(Float64, val))
        end

        # State 1 is the default state so geometry irrelevant
        if state_num > 1
            @match key begin
                "xmin" => (state.xmin = parse(Float64, val) + settings.dx / 100)
                "ymin" => (state.ymin = parse(Float64, val) + settings.dy / 100)
                "xmax" => (state.xmax = parse(Float64, val) - settings.dx / 100)
                "ymax" => (state.ymax = parse(Float64, val) - settings.dy / 100)
                "radius" => (state.radius = parse(Float64, val))
                "geometry" => (state.geometry = @match val begin
                    "rectangle" => Rectangular
                    "circular" => Circular
                    "point" => Point
                end)
            end
        end
    end

    return state
end

# Fetches the checking value from the test problems file
function checkingvalue(settings::Settings, problemfile = "tea.problems")::Float64
    open(problemfile, read = true) do file
        while !eof(file)
            thesplit = split(readline(file), " ")
            params = parse.(Int64, thesplit[1:3])
            checking_value = parse(Float64, thesplit[4])

            if params == [settings.xcells, settings.ycells, settings.endstep]
                # Found the problem in the file
                return checking_value
            end
        end

        @warn "Problem was not found in the test problems file."
        return 1.0
    end
end
