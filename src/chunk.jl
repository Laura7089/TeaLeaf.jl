export Chunk

# Empty extension point
@with_kw mutable struct ChunkExtension
    d_comm_buffer::Vector{Float64}
    d_reduce_buffer::Vector{Float64}
    d_reduce_buffer2::Vector{Float64}
    d_reduce_buffer3::Vector{Float64}
    d_reduce_buffer4::Vector{Float64}
end

@with_kw mutable struct Chunk
    @deftype Vector{Float64}
    # Solve-wide variables
    dt_init::Float64

    # Neighbouring ranks
    neighbours::Vector{Int} = Array{Int}(undef, NUM_FACES)

    # MPI comm buffers
    left_send::Any
    left_recv::Any
    right_send::Any
    right_recv::Any
    top_send::Any
    top_recv::Any
    bottom_send::Any
    bottom_recv::Any

    # Field dimensions
    x::Int
    y::Int

    # Mesh chunks
    left::Int
    right::Int = left + x
    bottom::Int
    top::Int = bottom + y

    # Field buffers
    density0 = Array{Float64}(undef, x * y)
    density = Array{Float64}(undef, x * y)
    energy0 = Array{Float64}(undef, x * y)
    energy = Array{Float64}(undef, x * y)

    u = Array{Float64}(undef, x * y)
    u0 = Array{Float64}(undef, x * y)
    p = Array{Float64}(undef, x * y)
    r = Array{Float64}(undef, x * y)
    mi = Array{Float64}(undef, x * y)
    w = Array{Float64}(undef, x * y)
    kx = Array{Float64}(undef, x * y)
    ky = Array{Float64}(undef, x * y)
    sd = Array{Float64}(undef, x * y)

    cell_x = Array{Float64}(undef, x)
    cell_y = Array{Float64}(undef, y)
    cell_dx = Array{Float64}(undef, x)
    cell_dy = Array{Float64}(undef, y)

    vertex_dx = Array{Float64}(undef, x + 1)
    vertex_dy = Array{Float64}(undef, y + 1)
    vertex_x = Array{Float64}(undef, x + 1)
    vertex_y = Array{Float64}(undef, y + 1)

    volume = Array{Float64}(undef, x * y)
    x_area = Array{Float64}(undef, (x + 1) * y)
    y_area = Array{Float64}(undef, x * (y + 1))

    # Cheby and PPCG
    # TODO: are these read outside of these solvers?
    theta::Float64 = 0.0
    eigmin::Float64 = 0.0
    eigmax::Float64 = 0.0

    cg_alphas::Any
    cg_betas::Any
    cheby_alphas::Any
    cheby_betas::Any

    ext::ChunkExtension
end

# Initialise the chunk
function Chunk(settings::Settings, x::Int, y::Int, left, bottom)::Chunk
    @info "Performing this solve with solver" settings.solver_name
    chunkx = x + settings.halo_depth * 2
    chunky = y + settings.halo_depth * 2

    lr_len = chunky * settings.halo_depth * NUM_FIELDS
    tb_len = chunkx * settings.halo_depth * NUM_FIELDS

    x_inner = chunkx - 2 * settings.halo_depth
    y_inner = chunky - 2 * settings.halo_depth

    return Chunk(
        # Initialise the key variables
        x = chunkx,
        y = chunky,
        dt_init = settings.dt_init,

        # Allocate the MPI comm buffers
        left_send = Array{Float64}(undef, lr_len),
        left_recv = Array{Float64}(undef, lr_len),
        right_send = Array{Float64}(undef, lr_len),
        right_recv = Array{Float64}(undef, lr_len),
        top_send = Array{Float64}(undef, tb_len),
        top_recv = Array{Float64}(undef, tb_len),
        bottom_send = Array{Float64}(undef, tb_len),
        bottom_recv = Array{Float64}(undef, tb_len),

        # Neighbours
        left = left,
        bottom = bottom,

        # Solver-specific
        cg_alphas = Array{Float64}(undef, settings.max_iters),
        cg_betas = Array{Float64}(undef, settings.max_iters),
        cheby_alphas = Array{Float64}(undef, settings.max_iters),
        cheby_betas = Array{Float64}(undef, settings.max_iters),

        # Initialise the ChunkExtension, which allows composition of extended
        # fields specific to individual implementations
        ext = ChunkExtension(
            d_comm_buffer = Array{Float64}(
                undef,
                settings.halo_depth * max(x_inner, y_inner),
            ),
            d_reduce_buffer = Array{Float64}(undef, chunkx * chunky),
            d_reduce_buffer2 = Array{Float64}(undef, chunkx * chunky),
            d_reduce_buffer3 = Array{Float64}(undef, chunkx * chunky),
            d_reduce_buffer4 = Array{Float64}(undef, chunkx * chunky),
        ),
    )
end

function set_chunk_data!(settings::Settings, chunk::Chunk)
    x_min = settings.grid_x_min + settings.dx * chunk.left
    y_min = settings.grid_y_min + settings.dy * chunk.bottom

    xᵢ = 2:chunk.x+1
    @. chunk.vertex_x[xᵢ] = x_min + settings.dx * (xᵢ - settings.halo_depth)
    yᵢ = 2:chunk.y+1
    @. chunk.vertex_y[yᵢ] = y_min + settings.dy * (yᵢ - settings.halo_depth)

    xᵢ = 2:chunk.x
    @. chunk.cell_x[xᵢ] = 0.5 * (chunk.vertex_x[xᵢ] + chunk.vertex_x[3:chunk.x+1])
    yᵢ = 2:chunk.y
    @. chunk.cell_y[yᵢ] = 0.5 * (chunk.vertex_y[yᵢ] + chunk.vertex_y[3:chunk.y+1])

    A = 2:chunk.x*chunk.y
    chunk.volume[A] .= settings.dx * settings.dy
    chunk.x_area[A] .= settings.dy
    chunk.y_area[A] .= settings.dx
end

function set_chunk_state!(chunk::Chunk, states::Vector{State})
    # Set the initial state
    chunk.energy0 .= states[1].energy
    chunk.density .= states[1].density

    # Apply all of the states in turn
    for ss in eachindex(states), jj = 1:chunk.y, kk = 1:chunk.x
        apply_state = @match states[ss].geometry begin
            Rectangular =>
                chunk.vertex_x[kk+1] >= states[ss].x_min &&
                    chunk.vertex_x[kk] < states[ss].x_max &&
                    chunk.vertex_y[jj+1] >= states[ss].y_min &&
                    chunk.vertex_y[jj] < states[ss].y_max
            Circular => begin
                radius = sqrt(
                    (chunk.cell_x[kk] - states[ss].x_min) *
                    (chunk.cell_x[kk] - states[ss].x_min) +
                    (chunk.cell_y[jj] - states[ss].y_min) *
                    (chunk.cell_y[jj] - states[ss].y_min),
                )
                radius <= states[ss].radius
            end
            Point =>
                chunk.vertex_x[kk] == states[ss].x_min &&
                    chunk.vertex_y[jj] == states[ss].y_min
        end

        # Check if state applies at this vertex, and apply
        if apply_state
            index = kk + jj * chunk.x
            chunk.energy0[index] = states[ss].energy
            chunk.density[index] = states[ss].density
        end
    end

    # Set an initial state for u
    index = @. (1:chunk.x-1) + (1:chunk.y-1) * chunk.x
    @. chunk.u[index] = chunk.energy0[index] * chunk.density[index]
end
