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

# TODO: fix all the indexing
# TODO: doc comments -> docstrings
# TODO: fix naming convention

# Global constants
@consts begin
    CG_ITERS_FOR_EIGENVALUES = 20 # TODO: still needed?
end

function main()
    settings = Settings()
    chunk = initialiseapp!(settings)
    diffuse!(chunk, settings)
end

function initialiseapp!(
    settings::Settings,
    useconfig = true,
    configfile = "tea.in",
    useflags = true,
)
    if useconfig
        readconfig!(settings, configfile)
    end
    if useflags
        parseflags!(settings)
    end

    states = readstates(settings)

    chunk = Chunk(settings)
    set_chunk_data!(settings, chunk)
    set_chunk_state!(chunk, states)

    # Prime the initial halo data
    # settings.toexchange .= false
    resettoexchange!(settings)
    settings.toexchange[:density] = true
    settings.toexchange[:energy0] = true
    settings.toexchange[:energy1] = true
    # TODO: is depth=1 correct here?
    haloupdate!(chunk, settings, 1)

    chunk.energy .= chunk.energy0

    return chunk
end

# The main timestep loop
function diffuse!(chunk::C, set::Settings) where {C<:Chunk}
    for tt = 1:set.endstep
        # Calculate minimum timestep information
        dt = min(chunk.dtinit, set.dtinit)

        rx = dt / set.dx^2
        ry = dt / set.dy^2

        # Prepare halo regions for solve
        resettoexchange!(set)
        set.toexchange[:energy1] = true
        set.toexchange[:density] = true
        # TODO: is depth=1 correct here?
        haloupdate!(chunk, set, 1)

        # Perform the solve with one of the integrated solvers
        error = set.solver.solve!(chunk, set, rx, ry)

        # Perform solve finalisation tasks
        solvefinished!(chunk, set)

        if tt % set.summaryfrequency == 0
            fieldsummary(chunk, set, false)
        end

        @info "Timestep $(tt) finished" set.solver error
    end
    fieldsummary(chunk, set, true)
end

end
