module TeaLeaf
export main

using Match
using ArgParse
using Parameters

include("./settings.jl")
include("./chunk.jl")
include("./initialise.jl")
include("./kernel.jl")
include("./drivers.jl")
include("./diffuse.jl")

# TODO: should all the `Vector{X}` parameters in this be views?
# TODO: all the 2 and 3 starting loop indices are _definitely_ not right
# TODO: replace all the stride-style indexing with julia matrices
# TODO: doc comments -> docstrings
# TODO: fix naming convention

# Global constants
@consts begin
    MASTER = 0

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

# Sparse Matrix Vector Product
function SMVP(chunk::Chunk, a::Vector{Float64}, index::Int64)::Float64
    (
        1.0 +
        (chunk.kx[index+1] + chunk.kx[index]) +
        (chunk.ky[index+chunk.x] + chunk.ky[index])
    ) * a[index]
    -(chunk.kx[index+1] * a[index+1] + chunk.kx[index] * a[index-1])
    -(chunk.ky[index+chunk.x] * a[index+chunk.x] + chunk.ky[index] * a[index-chunk.x])
end

function main()
    settings, chunks = initialise_application() # Done
    diffuse(chunks, settings) # Done
end

end
