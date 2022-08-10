export Chunk
export CHUNK_FIELDS
export CHUNK_LEFT, CHUNK_RIGHT, CHUNK_TOP, CHUNK_BOTTOM

@consts begin
    CHUNK_LEFT = 1
    CHUNK_RIGHT = 2
    CHUNK_BOTTOM = 3
    CHUNK_TOP = 4

    CHUNK_FIELDS = [:density, :p, :energy0, :energy, :u, :sd]

    NUM_FIELDS = 6
end

@with_kw mutable struct Chunk{T<:AbstractMatrix{Float64}}
    # Solve-wide variables
    dt_init::Float64

    # Field dimensions
    x::Int
    y::Int

    # Field buffers
    density0::T = Array{Float64}(undef, x, y)
    density::T = Array{Float64}(undef, x, y)
    energy0::T = Array{Float64}(undef, x, y)
    energy::T = Array{Float64}(undef, x, y)

    u::T = Array{Float64}(undef, x, y)
    u0::T = Array{Float64}(undef, x, y)
    p::T = Array{Float64}(undef, x, y)
    r::T = Array{Float64}(undef, x, y)
    mi::T = Array{Float64}(undef, x, y)
    w::T = Array{Float64}(undef, x, y)
    kx::T = Array{Float64}(undef, x, y)
    ky::T = Array{Float64}(undef, x, y)
    sd::T = Array{Float64}(undef, x, y)

    cell_x::AbstractVector{Float64} = Array{Float64}(undef, x)
    cell_y::AbstractVector{Float64} = Array{Float64}(undef, y)
    cell_dx::AbstractVector{Float64} = Array{Float64}(undef, x)
    cell_dy::AbstractVector{Float64} = Array{Float64}(undef, y)

    vertex_dx::AbstractVector{Float64} = Array{Float64}(undef, x + 1)
    vertex_dy::AbstractVector{Float64} = Array{Float64}(undef, y + 1)
    vertex_x::AbstractVector{Float64} = Array{Float64}(undef, x + 1)
    vertex_y::AbstractVector{Float64} = Array{Float64}(undef, y + 1)

    volume::T = Array{Float64}(undef, x, y)
    x_area::T = Array{Float64}(undef, x + 1, y)
    y_area::T = Array{Float64}(undef, x, y + 1)

    # Cheby and PPCG
    # TODO: are these read outside of these solvers?
    theta::Float64 = 0.0
    eigmin::Float64 = 0.0
    eigmax::Float64 = 0.0

    cg_alphas::Vector{Float64}
    cg_betas::Vector{Float64}
    cheby_alphas::Vector{Float64}
    cheby_betas::Vector{Float64}
end

# Initialise the chunk
function Chunk(settings::Settings)
    @info "Performing this solve with solver" settings.solver_name settings.solver
    chunkx = settings.grid_x_cells + settings.halo_depth * 2
    chunky = settings.grid_y_cells + settings.halo_depth * 2

    lr_len = chunky * settings.halo_depth * NUM_FIELDS
    tb_len = chunkx * settings.halo_depth * NUM_FIELDS

    x_inner = chunkx - 2 * settings.halo_depth
    y_inner = chunky - 2 * settings.halo_depth

    return Chunk(
        # Initialise the key variables
        x = chunkx,
        y = chunky,
        dt_init = settings.dt_init,

        # Solver-specific
        cg_alphas = Array{Float64}(undef, settings.max_iters),
        cg_betas = Array{Float64}(undef, settings.max_iters),
        cheby_alphas = Array{Float64}(undef, settings.max_iters),
        cheby_betas = Array{Float64}(undef, settings.max_iters),
    )
end

function set_chunk_data!(settings::Settings, chunk::Chunk)
    x_min = settings.grid_x_min + settings.dx
    y_min = settings.grid_y_min + settings.dy

    xᵢ = 2:chunk.x+1
    @. chunk.vertex_x[xᵢ] = x_min + settings.dx * (xᵢ - settings.halo_depth)
    yᵢ = 2:chunk.y+1
    @. chunk.vertex_y[yᵢ] = y_min + settings.dy * (yᵢ - settings.halo_depth)

    xᵢ = 2:chunk.x
    @. chunk.cell_x[xᵢ] = 0.5 * (chunk.vertex_x[xᵢ] + chunk.vertex_x[3:chunk.x+1])
    yᵢ = 2:chunk.y
    @. chunk.cell_y[yᵢ] = 0.5 * (chunk.vertex_y[yᵢ] + chunk.vertex_y[3:chunk.y+1])

    chunk.volume .= settings.dx * settings.dy
    chunk.x_area .= settings.dy
    chunk.y_area .= settings.dx
end

function set_chunk_state!(chunk::Chunk, states::Vector{State})
    # Set the initial state
    chunk.energy0 .= states[1].energy
    chunk.density .= states[1].density

    # Apply all of the states in turn
    for ss in eachindex(states), jj = 1:chunk.y, kk = 1:chunk.x
        apply_state = @match states[ss].geometry begin
            Rectangular => all((
                chunk.vertex_x[kk+1] >= states[ss].x_min,
                chunk.vertex_x[kk] < states[ss].x_max,
                chunk.vertex_y[jj+1] >= states[ss].y_min,
                chunk.vertex_y[jj] < states[ss].y_max,
            ))
            Circular => begin
                radius = sqrt(
                    (chunk.cell_x[kk] - states[ss].x_min)^2 +
                    (chunk.cell_y[jj] - states[ss].y_min)^2,
                )
                radius <= states[ss].radius
            end
            Point =>
                chunk.vertex_x[kk] == states[ss].x_min &&
                    chunk.vertex_y[jj] == states[ss].y_min
        end

        # Check if state applies at this vertex, and apply
        if apply_state
            chunk.energy0[jj, kk] = states[ss].energy
            chunk.density[jj, kk] = states[ss].density
        end
    end

    # Set an initial state for u
    # TODO: correct indexing?
    @. chunk.u[1:chunk.x-1, 1:chunk.y-1] =
        chunk.energy0[1:chunk.x-1, 1:chunk.y-1] * chunk.density[1:chunk.x-1, 1:chunk.y-1]
end
