export Chunk
export chunksize
export CHUNK_EXCHANGE_FIELDS
export halo, haloa

const CHUNK_EXCHANGE_FIELDS = [:density, :p, :energy0, :energy, :u, :sd]

@with_kw mutable struct Chunk{T<:AbstractMatrix{Float64}}
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
    chunkx = settings.xcells + 2settings.halodepth
    chunky = settings.ycells + 2settings.halodepth

    return Chunk(
        # Initialise the key variables
        x = chunkx,
        y = chunky,

        # Solver-specific
        cgα = zeros(settings.maxiters),
        cgβ = zeros(settings.maxiters),
        chebyα = zeros(settings.maxiters),
        chebyβ = zeros(settings.maxiters),
    )
end

Base.size(c::Chunk) = size(c.density0)
Base.axes(c::Chunk) = axes(c.density0)
haloa(c::Chunk, hd::Int) = Tuple(ax[begin+hd:end-hd] for ax in axes(c))
halo(c::Chunk, hd::Int) = CartesianIndices(haloa(c, hd))

function setchunkdata!(settings::Settings, chunk::Chunk)
    xₛ, yₛ = size(chunk)

    xᵢ = 1:xₛ+1
    @. chunk.vertexx[xᵢ] = settings.xmin + settings.dx * (xᵢ - settings.halodepth - 1)
    yᵢ = 1:yₛ+1
    @. chunk.vertexy[yᵢ] = settings.ymin + settings.dy * (yᵢ - settings.halodepth - 1)

    xᵢ, yᵢ = axes(chunk)
    @. chunk.cellx = 0.5(chunk.vertexx[xᵢ] + chunk.vertexx[xᵢ+1])
    @. chunk.celly = 0.5(chunk.vertexy[yᵢ] + chunk.vertexy[yᵢ+1])

    chunk.volume .= settings.dx * settings.dy
    chunk.xarea .= settings.dy
    chunk.yarea .= settings.dx
end

function setchunkstate!(chunk::Chunk, states::Vector{State})
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
    I = halo(chunk, 1) # note hardcoded 1
    @. chunk.u[I] = chunk.energy0[I] * chunk.density[I]
end
