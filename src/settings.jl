# The type of solver to be run
@enum Solver Jacobi CG Cheby PPCG

# The accepted types of state geometry
@enum Geometry Rectangular Circular Point

const CONDUCTIVITY = 1

# State list
struct State
    defined::Bool
    density::Float64
    energy::Float64
    x_min::Float64
    y_min::Float64
    x_max::Float64
    y_max::Float64
    radius::Float64
    geometry::Geometry
end

# The main settings structure
mutable struct Settings
    # Log files
    # tea_out_fp::String

    # Solve-wide constants
    rank::Int
    end_step::Int
    presteps::Int
    max_iters::Int
    coefficient::Int
    ppcg_inner_steps::Int
    summary_frequency::Int
    halo_depth::Int
    num_states::Int
    num_chunks::Int
    num_chunks_per_rank::Int
    num_ranks::Int
    fields_to_exchange::Vector{Bool}

    is_offload::Bool

    error_switch::Bool
    check_result::Bool
    preconditioner::Bool

    eps::Float64
    dt_init::Float64
    end_time::Float64
    eps_lim::Float64

    # Input-Output files
    tea_in_filename::String
    tea_out_filename::String
    test_problem_filename::String

    solver::Solver
    solver_name::String

    # Field dimensions
    grid_x_cells::Int
    grid_y_cells::Int

    grid_x_min::Float64
    grid_y_min::Float64
    grid_x_max::Float64
    grid_y_max::Float64

    dx::Float64
    dy::Float64
end

# TODO: this is really stupid
Settings() = Settings(
    0,
    typemax(Int),
    30,
    10_000,
    CONDUCTIVITY,
    10,
    10,
    1,
    0,
    1,
    1,
    1,
    Array{Bool}(undef, 6),
    false,
    0,
    1,
    0,
    1e-15,
    0.1,
    10.0,
    1e-5,
    "tea.in",
    "tea.out",
    "tea.problems",
    CG,
    "",
    10,
    10,
    0.0,
    0.0,
    100.0,
    100.0,
    0.0,
    0.0,
)

function reset_fields_to_exchange(settings::Settings)
    settings.fields_to_exchange .= false
end

function read_config(settings::Settings)
    # Open the configuration file
    open(settings.tea_in_filename, read = true) do tea_in
        while !eof(tea_in)
            thesplit = split(readline(tea_in), " ")
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
            end
        end

        # Read in the states
        states, settings.num_states = read_states(settings) # Done
    end

    @info "Solution Parameters" settings

    for ss = 0:settings.num_states
        @info "state $(ss)" ss states[ss].density states[ss].energy
        if ss > 0
            @info "" states[ss].x_min states[ss].y_min states[ss].x_max states[ss].y_max states[ss].radius states[ss].geometry
        end
    end
    return states
end

# Read all of the states from the configuration file
function read_states(settings::Settings)::Tuple(Vector{State}, Int)
    len = 0
    line = nothing
    num_states = 0

    # First find the number of states
    open(settings.tea_in_filename, read = true) do tea_in
        while !eof(tea_in)
            state_num = 0

            line = readline(tea_in)
            if startswith(line, "state")
                num_states = max(num_states, parse(Int, split(line)[2]))
            end
        end
    end

    # Pre-initialise the set of states
    states = Array{State}(undef, num_states)
    for ss in eachindex(states)
        states[ss].defined = false
    end

    # If a state boundary falls exactly on a cell boundary
    # then round off can cause the state to be put one cell
    # further than expected. This is compiler/system dependent.
    # To avoid this, a state boundary is reduced/increased by a
    # 100th of a cell width so it lies well within the intended
    # cell. Because a cell is either full or empty of a specified
    # state, this small modification to the state extents does
    # not change the answer.
    open(settings.tea_in_filename, read = true) do tea_in
        while !eof(tea_in)
            state_num = 0

            line = readline(tea_in)
            thesplit = split(line, " ")

            # State found
            if startswith("state", line)
                state_num = parse(Int, thesplit[2])
                state = states[state_num]

                if state.defined
                    throw("State number $(state_num) defined twice.")
                end

                read_value(line, "density", word)
                state.density = atof(word)
                read_value(line, "energy", word)
                state.energy = atof(word)

                for pair in split(line, " ")[3:end]
                    key = split(pair, "=")[0]
                    val = split(pair, "=")[1]

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
                state.defined = true
            end

            return (states, num_states)
        end
    end
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
