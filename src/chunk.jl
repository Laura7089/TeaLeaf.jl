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
    dtinit::Float64

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

    cellx::AbstractVector{Float64} = Array{Float64}(undef, x)
    celly::AbstractVector{Float64} = Array{Float64}(undef, y)
    celldx::AbstractVector{Float64} = Array{Float64}(undef, x)
    celldy::AbstractVector{Float64} = Array{Float64}(undef, y)

    vertexdx::AbstractVector{Float64} = Array{Float64}(undef, x + 1)
    vertexdy::AbstractVector{Float64} = Array{Float64}(undef, y + 1)
    vertexx::AbstractVector{Float64} = Array{Float64}(undef, x + 1)
    vertexy::AbstractVector{Float64} = Array{Float64}(undef, y + 1)

    volume::T = Array{Float64}(undef, x, y)
    xarea::T = Array{Float64}(undef, x + 1, y)
    yarea::T = Array{Float64}(undef, x, y + 1)

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

# Initialise the chunk
function Chunk(settings::Settings)
    @info "Performing this solve with solver" settings.solver
    chunkx = settings.xcells + settings.halodepth * 2
    chunky = settings.ycells + settings.halodepth * 2

    lr_len = chunky * settings.halodepth * NUM_FIELDS
    tb_len = chunkx * settings.halodepth * NUM_FIELDS

    x_inner = chunkx - 2 * settings.halodepth
    y_inner = chunky - 2 * settings.halodepth

    return Chunk(
        # Initialise the key variables
        x = chunkx,
        y = chunky,
        dtinit = settings.dtinit,

        # Solver-specific
        cgα = Array{Float64}(undef, settings.maxiters),
        cgβ = Array{Float64}(undef, settings.maxiters),
        chebyα = Array{Float64}(undef, settings.maxiters),
        chebyβ = Array{Float64}(undef, settings.maxiters),
    )
end

function set_chunk_data!(settings::Settings, chunk::Chunk)
    xmin = settings.xmin + settings.dx
    ymin = settings.ymin + settings.dy

    xᵢ = 2:chunk.x+1
    @. chunk.vertexx[xᵢ] = xmin + settings.dx * (xᵢ - settings.halodepth)
    yᵢ = 2:chunk.y+1
    @. chunk.vertexy[yᵢ] = ymin + settings.dy * (yᵢ - settings.halodepth)

    xᵢ = 2:chunk.x
    @. chunk.cellx[xᵢ] = 0.5 * (chunk.vertexx[xᵢ] + chunk.vertexx[3:chunk.x+1])
    yᵢ = 2:chunk.y
    @. chunk.celly[yᵢ] = 0.5 * (chunk.vertexy[yᵢ] + chunk.vertexy[3:chunk.y+1])

    chunk.volume .= settings.dx * settings.dy
    chunk.xarea .= settings.dy
    chunk.yarea .= settings.dx
end

function set_chunk_state!(chunk::Chunk, states::Vector{State})
    # Set the initial state
    chunk.energy0 .= states[1].energy
    chunk.density .= states[1].density

    # Apply all of the states in turn
    for ss in eachindex(states), jj = 1:chunk.y, kk = 1:chunk.x
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
    @. chunk.u[1:chunk.x-1, 1:chunk.y-1] =
        chunk.energy0[1:chunk.x-1, 1:chunk.y-1] * chunk.density[1:chunk.x-1, 1:chunk.y-1]
end
