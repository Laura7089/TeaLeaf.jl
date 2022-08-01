module TeaLeaf
export main

const FIELD_DENSITY = 0
const FIELD_ENERGY0 = 1
const FIELD_ENERGY1 = 2

include("./settings.jl")
include("./chunk.jl")
include("./initialise.jl")

function main()
    settings = Settings()
    chunks = Array{Chunk}(undef, settings.num_chunks_per_rank)
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
