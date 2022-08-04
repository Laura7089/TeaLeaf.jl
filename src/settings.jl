# The type of solver to be run
@enum Solver Jacobi CG Cheby PPCG

# The accepted types of state geometry
@enum Geometry Rectangular Circular Point

const CONDUCTIVITY = 1

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
    rank::Int = 0
    end_step::Int = typemax(Int)
    presteps::Int = 30
    max_iters::Int = 10_000
    coefficient::Int = CONDUCTIVITY
    ppcg_inner_steps::Int = 10
    summary_frequency::Int = 10
    halo_depth::Int = 1
    num_states::Int = 0
    num_chunks::Int = 1
    num_chunks_per_rank::Int = 1
    num_ranks::Int = 1
    fields_to_exchange::Vector{Bool} = Array{Bool}(undef, 6) # TODO

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

    solver::Solver = CG
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

function reset_fields_to_exchange(settings::Settings)
    settings.fields_to_exchange .= false
end

function read_config!(settings::Settings)
    states = []
    # Open the configuration file
    open(settings.tea_in_filename, read = true) do tea_in
        while !eof(tea_in)
            line = readline(tea_in)
            thesplit = split(line, " ")
            # Read all of the settings from the config
            @match first(thesplit) begin
                "initial_timestep" => (settings.dt_init = parse(Float64, thesplit[2]))
                "end_time" => (settings.end_time = parse(Float64, thesplit[2]))
                "end_step" => (settings.end_step = parse(Int, thesplit[2]))
                "xmin" => (settings.grid_x_min = parse(Float64, thesplit[2]))
                "ymin" => (settings.grid_y_min = parse(Float64, thesplit[2]))
                "xmax" => (settings.grid_x_max = parse(Float64, thesplit[2]))
                "ymax" => (settings.grid_y_max = parse(Float64, thesplit[2]))
                "x_cells",
                if settings.grid_x_cells == DEF_GRID_X_CELLS
                end => (settings.grid_x_cells = parse(Int, thesplit[2]))
                "y_cells",
                if settings.grid_y_cells == DEF_GRID_Y_CELLS
                end => (settings.grid_y_cells = parse(Int, thesplit[2]))
                "summary_frequency" =>
                    (settings.summary_frequency = parse(Int, thesplit[2]))
                "presteps" => (settings.presteps = parse(Int, thesplit[2]))
                "ppcg_inner_steps" => (settings.ppcg_inner_steps = parse(Int, thesplit[2]))
                "epslim" => (settings.eps_lim = parse(Float64, thesplit[2]))
                "max_iters" => (settings.max_iters = parse(Int, thesplit[2]))
                "eps" => (settings.eps = parse(Float64, thesplit[2]))
                "num_chunks_per_rank" =>
                    (settings.num_chunks_per_rank = parse(Int, thesplit[2]))
                "halo_depth" => (settings.halo_depth = parse(Int, thesplit[2]))
                "state" => (push!(states, read_state(line, settings)))
            end
        end
    end

    @info "Solution Parameters" settings

    for (ss, state) = enumerate(states)
        @info "state $(ss)" state.density state.energy state.x_min state.y_min state.x_max state.y_max state.radius state.geometry
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
                "xmin" =>
                (state.x_min = parse(Float64, val) + settings.dx / 100)
                "ymin" =>
                (state.y_min = parse(Float64, val) + settings.dy / 100)
                "xmax" =>
                (state.x_max = parse(Float64, val) - settings.dx / 100)
                "ymax" =>
                (state.y_max = parse(Float64, val) - settings.dy / 100)
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

function parse_flags!(settings::Settings)
    s = ArgParseSettings()

    @add_arg_table s begin
        "--solver", "-s"
        help = "Can be 'cg', 'cheby', 'ppcg', or 'jacobi'"
        arg_type = Solver
        "-x"
        arg_type = Int
        "-y"
        arg_type = Int
    end

    args = parse_args(s)
    if args["solver"] != Nothing
        settings.solver = args["solver"]
    end
    if args["x"] != Nothing
        settings.grid_x_cells = args["x"]
    end
    if args["y"] != Nothing
        settings.grid_y_cells = args["y"]
    end
end
