# The accepted types of state geometry
@enum Geometry Rectangular Circular Point

const CONDUCTIVITY = 1
const NUM_FIELDS = 6

# State list
@with_kw mutable struct State
    density::Float64 = 0.0
    energy::Float64 = 0.0
    x_min::Float64 = 0.0
    y_min::Float64 = 0.0
    x_max::Float64 = 0.0
    y_max::Float64 = 0.0
    radius::Float64 = 0.0
    geometry::Geometry = Rectangular
end

# The main settings structure
@with_kw mutable struct Settings
    # Solve-wide constants
    end_step::Int = typemax(Int)
    presteps::Int = 30
    max_iters::Int = 10_000
    coefficient::Int = CONDUCTIVITY
    ppcg_inner_steps::Int = 10
    summary_frequency::Int = 10
    halo_depth::Int = 1
    num_states::Int = 0
    fields_to_exchange::Vector{Bool} = fill(false, 6)

    is_offload::Bool = false

    error_switch::Bool = false
    check_result::Bool = true
    preconditioner::Bool = false

    eps::Float64 = 1e-15
    dt_init::Float64 = 0.1
    end_time::Float64 = 10.0
    eps_lim::Float64 = 1e-5

    # Input-Output files
    tea_in_filename::String = "tea.in"
    tea_out_filename::String = "tea.out"
    test_problem_filename::String = "tea.problems"

    solver::Module = CG
    solver_name::String = ""

    # Field dimensions
    grid_x_cells::Int = 10
    grid_y_cells::Int = 10

    grid_x_min::Float64 = 0.0
    grid_y_min::Float64 = 0.0
    grid_x_max::Float64 = 100.0
    grid_y_max::Float64 = 100.0

    dx::Float64 = 0.0
    dy::Float64 = 0.0
end
Broadcast.broadcastable(s::Settings) = Ref(s)

function read_config!(settings::Settings)::Vector{State}
    states = []
    # Open the configuration file
    @info "Reading configuration from $(settings.tea_in_filename)"
    open(settings.tea_in_filename, read = true) do tea_in
        while !eof(tea_in)
            line = readline(tea_in)
            @debug "Config line:" line

            # TODO: get test problems here too?
            if startswith(line, "state")
                push!(states, read_state(line, settings))
                continue
            elseif startswith(line, "*") || !occursin("=", line)
                continue
            end

            # TODO: rename all the fields that don't match?
            # Read all of the settings from the config
            key, val = split(line, "=")
            @match key begin
                "initial_timestep" => (settings.dt_init = parse(Float64, val))
                "xmin" => (settings.grid_x_min = parse(Float64, val))
                "ymin" => (settings.grid_y_min = parse(Float64, val))
                "xmax" => (settings.grid_x_max = parse(Float64, val))
                "ymax" => (settings.grid_y_max = parse(Float64, val))
                "tl_max_iters" => (settings.max_iters = parse(Float64, val))
                "tl_eps" => (settings.eps = parse(Float64, val))
                # TODO: call flag parsing after this so we don't have to check
                "x_cells",
                if settings.grid_x_cells == 10
                end => (settings.grid_x_cells = parse(Int, val))
                "y_cells",
                if settings.grid_y_cells == 10
                end => (settings.grid_y_cells = parse(Int, val))
                "epslim" => (settings.eps_lim = parse(Float64, val))
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

    for (ss, state) in enumerate(states)
        @debug "State $(ss)" state.density state.energy state.x_min state.y_min state.x_max state.y_max state.radius state.geometry
    end

    settings.num_states = length(states)
    return states
end

function read_state(line, settings::Settings)::State
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
                "xmin" => (state.x_min = parse(Float64, val) + settings.dx / 100)
                "ymin" => (state.y_min = parse(Float64, val) + settings.dy / 100)
                "xmax" => (state.x_max = parse(Float64, val) - settings.dx / 100)
                "ymax" => (state.y_max = parse(Float64, val) - settings.dy / 100)
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

function parse_flags()::Settings
    settings = Settings()
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
        settings.grid_x_cells = args["x"]
    end
    if !isnothing(args["y"])
        settings.grid_y_cells = args["y"]
    end
    return settings
end

# Fetches the checking value from the test problems file
function get_checking_value(settings::Settings)::Float64
    open(settings.test_problem_filename, read = true) do file
        # Get the number of states present in the config file
        while !eof(file)
            thesplit = split(readline(file), " ")
            params = parse.(Int64, thesplit[1:3])
            checking_value = parse(Float64, thesplit[4])

            if params == [settings.grid_x_cells, settings.grid_y_cells, settings.end_step]
                # Found the problem in the file
                return checking_value
            end
        end

        @warn "Problem was not found in the test problems file."
        return 1.0
    end
end
