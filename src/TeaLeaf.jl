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
    settings = Settings()
    states = read_config!(settings)
    parseflags!(settings)

    chunk = Chunk(settings)
    set_chunk_data!(settings, chunk)
    set_chunk_state!(chunk, states)

    # Prime the initial halo data
    # settings.toexchange .= false
    setindex!.(Ref(settings.toexchange), false, CHUNK_FIELDS)
    settings.toexchange[:density] = true
    settings.toexchange[:energy0] = true
    settings.toexchange[:energy1] = true
    # TODO: is depth=1 correct here?
    haloupdate!(chunk, settings, 1)

    store_energy!(chunk)

    return (settings, chunk)
end

# The main timestep loop
function diffuse!(chunk::C, set::Settings) where {C<:Chunk}
    for tt = 1:set.endstep
        # Calculate minimum timestep information
        dt = min(chunk.dtinit, set.dtinit)

        rx = dt / (set.dx * set.dx)
        ry = dt / (set.dy * set.dy)

        # Prepare halo regions for solve
        setindex!.(Ref(set.toexchange), false, CHUNK_FIELDS)
        set.toexchange[:energy1] = true
        set.toexchange[:density] = true
        # TODO: is depth=1 correct here?
        haloupdate!(chunk, set, 1)

        # Perform the solve with one of the integrated solvers
        error = set.solver.driver!(chunk, set, rx, ry)

        # Perform solve finalisation tasks
        solvefinished!(chunk, set)

        if tt % set.summaryfrequency == 0
            fieldsummary(chunk, set, false)
        end

        @info "Timestep finished" tt error
    end
    fieldsummary(chunk, set, true)
end

end
