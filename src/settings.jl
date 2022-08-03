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
