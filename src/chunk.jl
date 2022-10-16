using Match
using Parameters

export Chunk
export chunksize
export EXCHANGE_FIELDS
export halo, haloa

"""
All fields which can be exchanged by the program.
"""
const EXCHANGE_FIELDS = [:density, :p, :energy0, :energy, :u, :sd]

"""
Describes the current 2D state of the model.

See also [`initialiseapp!`](@ref), [`setchunkstate!`](@ref).
"""
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

    vertexx::AbstractVector{Float64}
    vertexy::AbstractVector{Float64}

    cellx::AbstractVector{Float64} = @. 0.5(vertexx[begin:end-1] + vertexx[begin+1:end])
    celly::AbstractVector{Float64} = @. 0.5(vertexy[begin:end-1] + vertexx[begin+1:end])

    volume::T = zeros(x, y)
    xarea::T = zeros(x + 1, y)
    yarea::T = zeros(x, y + 1)

    # Cheby and PPCG
    # TODO: are these read outside of these solvers?
    θ::Float64 = 0.0
    eigmin::Float64 = 0.0
    eigmax::Float64 = 0.0

    cgα::AbstractVector{Float64}
    cgβ::AbstractVector{Float64}
    chebyα::AbstractVector{Float64}
    chebyβ::AbstractVector{Float64}
end
Broadcast.broadcastable(c::Chunk) = Ref(c)

"""
    Chunk(settings)

Initialise an empty chunk from `settings`.
"""
function Chunk(set::Settings)
    x = set.xcells + 2set.halodepth
    y = set.ycells + 2set.halodepth

    return Chunk(
        x = x,
        y = y,

        vertexx = (@. set.xmin + set.dx * ((1:x+1) - set.halodepth - 1)),
        vertexy = (@. set.ymin + set.dy * ((1:y+1) - set.halodepth - 1)),

        volume = fill(set.dx * set.dy, (x, y)),
        xarea = fill(set.dy, (x, y)),
        yarea = fill(set.dx, (x, y)),

        # Solver-specific
        cgα = zeros(set.maxiters),
        cgβ = zeros(set.maxiters),
        chebyα = zeros(set.maxiters),
        chebyβ = zeros(set.maxiters),
    )
end

Base.size(c::Chunk) = size(c.density0)
Base.axes(c::Chunk) = axes(c.density0)

"""
    haloa(chunk, depth)

Get a [`Tuple`](@ref) of axes within `chunk` that represent the halo of depth `depth`.
"""
haloa(c::Chunk, hd::Int) = Tuple(ax[begin+hd:end-hd] for ax in axes(c))
"""
    halo(chunk, depth)

Convenience wrapper for [`CartesianIndices`](@ref) over [`haloa`](@ref).

# Examples

Use to get a 2D index that will work seamlessly in julia:

```julia-repl
julia> set = Settings("tea.in")
julia> chunk = initialiseapp!(set)
julia> chunk[halo(chunk, 2)]
```
"""
halo(c::Chunk, hd::Int) = CartesianIndices(haloa(c, hd))

"""
    setchunkstate!(chunk, states)

Apply each member of `states` in turn to `chunk`.
"""
function setchunkstate!(ch::Chunk, states::Vector{State})
    # Set the initial state
    ch.energy0 .= states[1].energy
    ch.density .= states[1].density

    x, y = size(ch)

    # Apply all of the states in turn
    for s in states, j = 1:y, k = 1:x
        apply = @match s.geometry begin
            Rectangular => all((
                ch.vertexx[k+1] >= s.xmin,
                ch.vertexx[k] < s.xmax,
                ch.vertexy[j+1] >= s.ymin,
                ch.vertexy[j] < s.ymax,
            ))
            Circular =>
                (ch.cellx[k] - s.xmin)^2 + (ch.celly[j] - s.ymin)^2 <= s.radius^2
            Point => ch.vertexx[k] == s.xmin && ch.vertexy[j] == s.ymin
        end
        apply || continue

        ch.energy0[j, k] = s.energy
        ch.density[j, k] = s.density
    end

    # Set an initial state for u
    I = halo(ch, 1) # note hardcoded 1
    @. ch.u[I] = ch.energy0[I] * ch.density[I]
end
