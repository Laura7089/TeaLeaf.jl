const FieldBufferType = Float64;

# Empty extension point
struct ChunkExtension
    d_comm_buffer::Vector{Float64}
    d_reduce_buffer::Vector{Float64}
    d_reduce_buffer2::Vector{Float64}
    d_reduce_buffer3::Vector{Float64}
    d_reduce_buffer4::Vector{Float64}
end

struct Chunk
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
