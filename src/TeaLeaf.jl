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

using TeaLeaf.Kernels

# TODO: all the 2 and 3 starting loop indices are _definitely_ not right
# TODO: doc comments -> docstrings
# TODO: fix naming convention

# Global constants
@consts begin
    CG_ITERS_FOR_EIGENVALUES = 20 # TODO: still needed?
end

function main()
    settings, chunk = initialise_application()
    diffuse!(chunk, settings)
end

function initialise_application()
    settings = parse_flags()
    states = read_config!(settings)

    chunk = Chunk(settings)
    set_chunk_data!(settings, chunk)
    set_chunk_state!(chunk, states)

    # Prime the initial halo data
    # settings.fields_to_exchange .= false
    setindex!.(Ref(settings.fields_to_exchange), false, CHUNK_FIELDS)
    settings.fields_to_exchange[:density] = true
    settings.fields_to_exchange[:energy0] = true
    settings.fields_to_exchange[:energy1] = true
    # TODO: is depth=1 correct here?
    halo_update!(chunk, settings, 1)

    store_energy!(chunk)

    return (settings, chunk)
end

# The main timestep loop
function diffuse!(chunk::C, set::Settings) where {C<:Chunk}
    for tt = 1:set.end_step
        # Calculate minimum timestep information
        dt = min(chunk.dt_init, set.dt_init)

        rx = dt / (set.dx * set.dx)
        ry = dt / (set.dy * set.dy)

        # Prepare halo regions for solve
        setindex!.(Ref(set.fields_to_exchange), false, CHUNK_FIELDS)
        set.fields_to_exchange[:energy1] = true
        set.fields_to_exchange[:density] = true
        # TODO: is depth=1 correct here?
        halo_update!(chunk, set, 1)

        # Perform the solve with one of the integrated solvers
        error = set.solver.driver!(chunk, set, rx, ry)

        # Perform solve finalisation tasks
        solve_finished!(chunk, set)

        if tt % set.summary_frequency == 0
            field_summary(chunk, set, false)
        end

        @info "Timestep finished" tt error
    end
    field_summary(chunk, set, true)
end

end
