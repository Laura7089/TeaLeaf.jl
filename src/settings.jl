export Settings
export CONDUCTIVITY, RECIP_CONDUCTIVITY
export checkingvalue
export resettoexchange!

# The accepted types of state geometry
@enum Geometry Rectangular Circular Point

const CONDUCTIVITY = 1
const RECIP_CONDUCTIVITY = 2

# State list
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
    ppcg_inner_steps::Int = 10
    summaryfrequency::Int = 10
    halodepth::Int = 2
    # Fields to exchange
    toexchange::Dict{Symbol,Bool} = Dict(zip(CHUNK_FIELDS, fill(false, 6)))

    is_offload::Bool = false

    errorswitch::Bool = false
    checkresult::Bool = true
    preconditioner::Bool = false

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
end
Broadcast.broadcastable(s::Settings) = Ref(s)
resettoexchange!(s::Settings) = setindex!.(Ref(s.toexchange), false, CHUNK_FIELDS)

function readconfig!(settings::Settings, infile)
    # Open the configuration file
    @info "Reading configuration from $(infile)"
    open(infile, read = true) do tea_in
        while !eof(tea_in)
            line = readline(tea_in)
            @debug "Config line:" line

            # TODO: get test problems here too?
            if startswith(line, "state") || startswith(line, "*") || !occursin("=", line)
                continue
            end

            # TODO: rename all the fields that don't match?
            # Read all of the settings from the config
            key, val = split(line, "=")
            @match key begin
                "initial_timestep" => (settings.dtinit = parse(Float64, val))
                # Use other keys as direct field names
                _ => try
                    key = Symbol(key)
                    val = parse(fieldtype(Settings, key), val)
                    setfield!(settings, key, val)
                catch
                    @warn "Unknown setting" line
                end
            end
        end
    end

    @info "Solution Parameters" settings
end

function readstates(settings::Settings, infile = "tea.in")::Vector{State}
    states = []
    # Open the configuration file
    @info "Reading state configuration from $(infile)"
    open(infile, read = true) do tea_in
        while !eof(tea_in)
            line = readline(tea_in)
            @debug "Config line:" line

            # TODO: get test problems here too?
            if !startswith(line, "state") || startswith(line, "*") || !occursin("=", line)
                continue
            end

            push!(states, readstate(line, settings))
        end
    end

    for (ss, state) in enumerate(states)
        @debug "State $(ss)" state.density state.energy state.xmin state.ymin state.x_max state.ymax state.radius state.geometry
    end
    return states
end

function readstate(line, settings::Settings)::State
    thesplit = split(line, " ")

    state_num = parse(Int, thesplit[2])
    state = State()

    for pair in split(line, " ")[3:end]
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

function parseflags!(settings::Settings)
    @info "Parsing flags..."
    s = ArgParseSettings()

    @add_arg_table s begin
        "--solver", "-s"
        help = "Can be 'cg', 'cheby', 'ppcg', or 'jacobi'"
        arg_type = Module
        "-x"
        arg_type = Int
        "-y"
        arg_type = Int
    end

    args = parse_args(s)
    if !isnothing(args["solver"])
        @match lowercase(args["solver"]) begin
            "jacobi" => (settings.solver = Jacobi)
            "cg" => (settings.solver = CG)
            "cheby" => (settings.solver = Cheby)
            "ppcg" => (settings.solver = PPCG)
        end
    end
    if !isnothing(args["x"])
        settings.xcells = args["x"]
    end
    if !isnothing(args["y"])
        settings.ycells = args["y"]
    end
end

# Fetches the checking value from the test problems file
function checkingvalue(settings::Settings, problemfile = "tea.problems")::Float64
    open(problemfile, read = true) do file
        # Get the number of states present in the config file
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
