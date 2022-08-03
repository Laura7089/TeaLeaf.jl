module TeaLeaf
export main

# Global constants
const MASTER = 0

const NUM_FACES = 4
const CHUNK_LEFT = 0
const CHUNK_RIGHT = 1
const CHUNK_BOTTOM = 2
const CHUNK_TOP = 3
const EXTERNAL_FACE = -1

const FIELD_DENSITY = 0
const FIELD_ENERGY0 = 1
const FIELD_ENERGY1 = 2
const FIELD_U = 3
const FIELD_P = 4
const FIELD_SD = 5

const CONDUCTIVITY = 1
const RECIP_CONDUCTIVITY = 2

const CG_ITERS_FOR_EIGENVALUES = 20
const ERROR_SWITCH_MAX = 1.0

include("./settings.jl")
include("./chunk.jl")
include("./initialise.jl")
include("./kernel.jl")

function main()
    settings = Settings() # Done
    chunks = Array{Chunk}(undef, settings.num_chunks_per_rank) # Done
    initialise_application(chunks, settings)
    settings_overload(settings, argc, argv)
    diffuse(chunks, settings)
    kernel_finalise_driver(chunks, settings)

    # Finalise each individual chunk
    for cc = 0:settings.num_chunks_per_rank
        finalise_chunk(chunks[cc])
    end
end

end
