export Chunk
export chunksize
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
    dtinit::Float64

    # Field dimensions
    x::Int
    y::Int

    # Field buffers
    density0::T = zeros(x, y)
    density::T = zeros(x, y)
    energy0::T = zeros(x, y)
    energy::T = zeros(x, y)

    u::T = zeros(x, y)
    u0::T = zeros(x, y)
    p::T = zeros(x, y)
    r::T = zeros(x, y)
    mi::T = zeros(x, y)
    w::T = zeros(x, y)
    kx::T = zeros(x, y)
    ky::T = zeros(x, y)
    sd::T = zeros(x, y)

    cellx::AbstractVector{Float64} = zeros(x)
    celly::AbstractVector{Float64} = zeros(y)
    celldx::AbstractVector{Float64} = zeros(x)
    celldy::AbstractVector{Float64} = zeros(y)

    vertexdx::AbstractVector{Float64} = zeros(x + 1)
    vertexdy::AbstractVector{Float64} = zeros(y + 1)
    vertexx::AbstractVector{Float64} = zeros(x + 1)
    vertexy::AbstractVector{Float64} = zeros(y + 1)

    volume::T = zeros(x, y)
    xarea::T = zeros(x + 1, y)
    yarea::T = zeros(x, y + 1)

    # Cheby and PPCG
    # TODO: are these read outside of these solvers?
    θ::Float64 = 0.0
    eigmin::Float64 = 0.0
    eigmax::Float64 = 0.0

    cgα::Vector{Float64}
    cgβ::Vector{Float64}
    chebyα::Vector{Float64}
    chebyβ::Vector{Float64}
end
Broadcast.broadcastable(c::Chunk) = Ref(c)

# Initialise the chunk
function Chunk(settings::Settings)
    @info "Performing this solve with solver" settings.solver
    chunkx = settings.xcells + 2settings.halodepth
    chunky = settings.ycells + 2settings.halodepth

    lr_len = chunky * settings.halodepth * NUM_FIELDS
    tb_len = chunkx * settings.halodepth * NUM_FIELDS

    x_inner = chunkx - 2settings.halodepth
    y_inner = chunky - 2settings.halodepth

    return Chunk(
        # Initialise the key variables
        x = chunkx,
        y = chunky,
        dtinit = settings.dtinit,

        # Solver-specific
        cgα = zeros(settings.maxiters),
        cgβ = zeros(settings.maxiters),
        chebyα = zeros(settings.maxiters),
        chebyβ = zeros(settings.maxiters),
    )
end

Base.size(c::Chunk) = size(c.density0)

function set_chunk_data!(settings::Settings, chunk::Chunk)
    xmin = settings.xmin + settings.dx
    ymin = settings.ymin + settings.dy
    xₛ, yₛ = size(chunk)

    xᵢ = 2:xₛ+1
    @. chunk.vertexx[xᵢ] = xmin + settings.dx * (xᵢ - settings.halodepth)
    yᵢ = 2:yₛ+1
    @. chunk.vertexy[yᵢ] = ymin + settings.dy * (yᵢ - settings.halodepth)

    xᵢ = 2:xₛ
    @. chunk.cellx[xᵢ] = 0.5(chunk.vertexx[xᵢ] + chunk.vertexx[3:xₛ+1])
    yᵢ = 2:yₛ
    @. chunk.celly[yᵢ] = 0.5(chunk.vertexy[yᵢ] + chunk.vertexy[3:yₛ+1])

    chunk.volume .= settings.dx * settings.dy
    chunk.xarea .= settings.dy
    chunk.yarea .= settings.dx
end

function set_chunk_state!(chunk::Chunk, states::Vector{State})
    # Set the initial state
    chunk.energy0 .= states[1].energy
    chunk.density .= states[1].density

    x, y = size(chunk)

    # Apply all of the states in turn
    for ss in eachindex(states), jj = 1:y, kk = 1:x
        apply_state = @match states[ss].geometry begin
            Rectangular => all((
                chunk.vertexx[kk+1] >= states[ss].xmin,
                chunk.vertexx[kk] < states[ss].xmax,
                chunk.vertexy[jj+1] >= states[ss].ymin,
                chunk.vertexy[jj] < states[ss].ymax,
            ))
            Circular => begin
                radius = sqrt(
                    (chunk.cellx[kk] - states[ss].xmin)^2 +
                    (chunk.celly[jj] - states[ss].ymin)^2,
                )
                radius <= states[ss].radius
            end
            Point =>
                chunk.vertexx[kk] == states[ss].xmin &&
                    chunk.vertexy[jj] == states[ss].ymin
        end

        # Check if state applies at this vertex, and apply
        if apply_state
            chunk.energy0[jj, kk] = states[ss].energy
            chunk.density[jj, kk] = states[ss].density
        end
    end

    # Set an initial state for u
    # TODO: correct indexing?
    @. chunk.u[1:x-1, 1:y-1] = chunk.energy0[1:x-1, 1:y-1] * chunk.density[1:x-1, 1:y-1]
end
