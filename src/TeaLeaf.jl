module TeaLeaf
export main

using Match
using ArgParse
using Parameters

# TODO: should all the `Vector{X}` parameters in this be views?
# TODO: all the 2 and 3 starting loop indices are _definitely_ not right
# TODO: replace all the stride-style indexing with julia matrices
# TODO: doc comments -> docstrings

# Global constants
const MASTER = 0

const NUM_FACES = 4
const CHUNK_LEFT = 1
const CHUNK_RIGHT = 2
const CHUNK_BOTTOM = 3
const CHUNK_TOP = 4
const EXTERNAL_FACE = -1

const FIELD_DENSITY = 1
const FIELD_ENERGY0 = 2
const FIELD_ENERGY1 = 3
const FIELD_U = 4
const FIELD_P = 5
const FIELD_SD = 6

const CONDUCTIVITY = 1
const RECIP_CONDUCTIVITY = 2

const CG_ITERS_FOR_EIGENVALUES = 20
const ERROR_SWITCH_MAX = 1.0

include("./settings.jl")
include("./chunk.jl")
include("./initialise.jl")
include("./kernel.jl")
include("./drivers.jl")
include("./diffuse.jl")

function main()
    settings, chunks = initialise_application() # Done
    diffuse(chunks, settings) # Done
end

end
