module TeaLeaf
export main, initialiseapp!, diffuse!

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
    @info "Solution Parameters" settings
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
    setchunkdata!(settings, chunk)
    setchunkstate!(chunk, states)

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
    if set.debugfile != "" && isfile(set.debugfile)
        rm(set.debugfile)
    end

    for tt = 1:set.endstep
        debugrecord(set, chunk)
        rx = set.dtinit / set.dx^2
        ry = set.dtinit / set.dy^2

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
            fieldsummary(chunk, set)
        end

        @info "Timestep $(tt) finished"
    end
    fieldsummary(chunk, set)
end

function debugrecord(settings::Settings, chunk::C) where {C<:Chunk}
    if settings.debugfile == ""
        return
    end

    @info "Writing debug data to $(settings.debugfile)"
    open(settings.debugfile, write = true, append = true) do debugfile
        for attr in filter(f -> !(f in (:x, :y)), fieldnames(Chunk))
            println(debugfile, String(attr))
            println(debugfile, getstring(getfield(chunk, attr)))
            println(debugfile, "")
        end
        println(debugfile, "")
        println(debugfile, "")
    end
end

function getstring(a::AbstractArray{T})::String where {T<:Number}
    join(join.(eachcol(a), ' '), '\n')
end
getstring(f) = string(f)

end
