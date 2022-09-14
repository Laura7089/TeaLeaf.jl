module TeaLeaf
export initialiseapp!, diffuse!

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

function initialiseapp!(settings::Settings)
    chunk = Chunk(settings)
    setchunkstate!(chunk, settings.states)
    # Prime the initial halo data
    haloupdate!(chunk, settings, 1, [:density, :energy0, :energy]) # TODO: is depth=1 correct

    chunk.energy .= chunk.energy0

    return chunk
end

# The main timestep loop
function diffuse!(chunk::Chunk, set::Settings)
    if set.debugfile != "" && isfile(set.debugfile)
        rm(set.debugfile)
    end

    for tt = 1:set.endstep
        debugrecord(set, chunk)
        rx = set.dtinit / set.dx^2
        ry = set.dtinit / set.dy^2
        haloupdate!(chunk, set, 1, [:energy, :density]) # TODO: is depth=1 correct

        # Perform the solve with one of the integrated solvers
        error = set.solver.solve!(chunk, set, rx, ry)

        # Perform solve finalisation tasks
        solvefinished!(chunk, set)

        tt % set.summaryfrequency == 0 && fieldsummary(chunk, set)
        @info "Timestep $(tt) finished"
    end
    fieldsummary(chunk, set)
end

function debugrecord(settings::Settings, chunk::Chunk)
    settings.debugfile == "" && return

    @info "Writing debug data to $(settings.debugfile)"
    open(settings.debugfile, write = true, append = true) do debugfile
        for attr in filter(f -> !(f in (:x, :y)), fieldnames(Chunk))
            println(debugfile, String(attr))
            println(debugfile, getstring(getfield(chunk, attr)))
            println(debugfile, "")
        end
        print(debugfile, "\n\n")
    end
end

function getstring(a::AbstractArray{T})::String where {T<:Number}
    join(join.(eachcol(a), ' '), '\n')
end
getstring(f) = string(f)

end
