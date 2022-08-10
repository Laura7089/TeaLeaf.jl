module TeaLeaf
export main

using Match
using ArgParse
using Parameters

include("./settings.jl")
include("./chunk.jl")
include("./kernels.jl")

include("./solvers/CG.jl")
include("./solvers/Cheby.jl")
include("./solvers/Jacobi.jl")
include("./solvers/PPCG.jl")

# TODO: all the 2 and 3 starting loop indices are _definitely_ not right
# TODO: replace all the stride-style indexing with julia matrices
# TODO: doc comments -> docstrings
# TODO: fix naming convention

# Global constants
@consts begin
    MASTER = 0

    ERROR_START = 1e+10

    NUM_FACES = 4
    CHUNK_LEFT = 1
    CHUNK_RIGHT = 2
    CHUNK_BOTTOM = 3
    CHUNK_TOP = 4
    EXTERNAL_FACE = -1

    FIELD_DENSITY = 1
    FIELD_ENERGY0 = 2
    FIELD_ENERGY1 = 3
    FIELD_U = 4
    FIELD_P = 5
    FIELD_SD = 6

    CONDUCTIVITY = 1
    RECIP_CONDUCTIVITY = 2

    CG_ITERS_FOR_EIGENVALUES = 20
    ERROR_SWITCH_MAX = 1.0
end

function main()
    settings, chunks = initialise_application() # Done
    diffuse!(chunks, settings) # Done
end

function initialise_application()
    settings = parse_flags()
    states = read_config!(settings) # Done

    chunk = Chunk(settings)
    set_chunk_data!(settings, chunk) # Done
    set_chunk_state!(chunk, states) # Done

    # Prime the initial halo data
    settings.fields_to_exchange .= false
    settings.fields_to_exchange[FIELD_DENSITY] = true
    settings.fields_to_exchange[FIELD_ENERGY0] = true
    settings.fields_to_exchange[FIELD_ENERGY1] = true
    # TODO: is depth=1 correct here?
    halo_update!(chunk, settings, 1) # Done

    store_energy!(chunk) # Done

    return (settings, chunk)
end

# The main timestep loop
function diffuse!(chunk::C, settings::Settings) where {C<:Chunk}
    for tt = 1:settings.end_step
        # Calculate minimum timestep information
        dt = min(chunk.dt_init, settings.dt_init)

        rx = dt / (settings.dx * settings.dx)
        ry = dt / (settings.dy * settings.dy)

        # Prepare halo regions for solve
        settings.fields_to_exchange .= false
        settings.fields_to_exchange[FIELD_ENERGY1] = true
        settings.fields_to_exchange[FIELD_DENSITY] = true
        # TODO: is depth=1 correct here?
        halo_update!(chunk, settings, 1) # Done

        # Perform the solve with one of the integrated solvers
        error = settings.solver.driver!(chunk, settings, rx, ry)

        # Perform solve finalisation tasks
        solve_finished!(chunk, settings) # Done

        if tt % settings.summary_frequency == 0
            field_summary(chunk, settings, false) # Done
        end

        @info "Timestep finished" tt error
    end
    field_summary(chunk, settings, true) # Done
end

end
