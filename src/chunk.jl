const FieldBufferType = Float64

# Empty extension point
struct ChunkExtension
    d_comm_buffer::Vector{Float64}
    d_reduce_buffer::Vector{Float64}
    d_reduce_buffer2::Vector{Float64}
    d_reduce_buffer3::Vector{Float64}
    d_reduce_buffer4::Vector{Float64}
end

mutable struct Chunk
    # Solve-wide variables
    dt_init::Float64

    # Neighbouring ranks
    neighbours::Vector{Int}

    # MPI comm buffers
    left_send::Vector{Float64}
    left_recv::Vector{Float64}
    right_send::Vector{Float64}
    right_recv::Vector{Float64}
    top_send::Vector{Float64}
    top_recv::Vector{Float64}
    bottom_send::Vector{Float64}
    bottom_recv::Vector{Float64}

    # Mesh chunks
    left::Int
    right::Int
    bottom::Int
    top::Int

    # Field dimensions
    x::Int
    y::Int

    # Field buffers
    density0::FieldBufferType
    density::FieldBufferType
    energy0::FieldBufferType
    energy::FieldBufferType

    u::FieldBufferType
    u0::FieldBufferType
    p::FieldBufferType
    r::FieldBufferType
    mi::FieldBufferType
    w::FieldBufferType
    kx::FieldBufferType
    ky::FieldBufferType
    sd::FieldBufferType

    cell_x::FieldBufferType
    cell_y::FieldBufferType
    cell_dx::FieldBufferType
    cell_dy::FieldBufferType

    vertex_dx::FieldBufferType
    vertex_dy::FieldBufferType
    vertex_x::FieldBufferType
    vertex_y::FieldBufferType

    volume::FieldBufferType
    x_area::FieldBufferType
    y_area::FieldBufferType

    # Cheby and PPCG
    theta::Float64
    eigmin::Float64
    eigmax::Float64

    cg_alphas::Vector{Float64}
    cg_betas::Vector{Float64}
    cheby_alphas::Vector{Float64}
    cheby_betas::Vector{Float64}

    ext::Vector{ChunkExtension}
end

# Initialise the chunk
function Chunk(settings::Settings, x::Int, y::Int)
    chunk = Chunk()
    # Initialise the key variables
    chunk.x = x + settings.halo_depth * 2
    chunk.y = y + settings.halo_depth * 2
    chunk.dt_init = settings.dt_init

    # Allocate the neighbour list
    chunk.neighbours = Array{int}(undef, NUM_FACES)

    # Allocate the MPI comm buffers
    lr_len = chunk.y * settings.halo_depth * NUM_FIELDS
    chunk.left_send = Array{double}(undef, lr_len)
    chunk.left_recv = Array{double}(undef, lr_len)
    chunk.right_send = Array{double}(undef, lr_len)
    chunk.right_recv = Array{double}(undef, lr_len)

    tb_len = chunk.x * settings.halo_depth * NUM_FIELDS
    chunk.top_send = Array{double}(undef, tb_len)
    chunk.top_recv = Array{double}(undef, tb_len)
    chunk.bottom_send = Array{double}(undef, tb_len)
    chunk.bottom_recv = Array{double}(undef, tb_len)

    # Initialise the ChunkExtension, which allows composition of extended
    # fields specific to individual implementations
    chunk.ext = ChunkExtension()
    return chunk
end
