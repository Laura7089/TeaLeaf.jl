module CG

using TeaLeaf
using TeaLeaf.Kernels
using LinearAlgebra: norm

# Performs a full solve with the CG solver kernels
function solve!(chunk::C, set::Settings, rx::Float64, ry::Float64)::Float64 where {C<:Chunk}
    # Perform CG initialisation
    rro = init!(chunk, set.halodepth, set.coefficient, rx, ry)

    resettoexchange!(set)
    set.toexchange[:u] = true
    set.toexchange[:p] = true
    haloupdate!(chunk, set, 1)

    copyu!(chunk, set.halodepth)

    error = ERROR_START

    iters = 0
    # Iterate till convergence
    for tt = 1:set.maxiters
        iters = tt
        rro = error = mainstep!(chunk, set, tt, rro)

        haloupdate!(chunk, set, 1)

        if sqrt(abs(error)) < set.eps
            break
        end
    end

    @info "CG solve complete" iters error
    return error
end

# Invokes the main CG solve kernels
function mainstep!(
    chunk::C,
    settings::Settings,
    tt::Int,
    rro::Float64,
)::Float64 where {C<:Chunk}
    pw = w!(chunk, settings.halodepth)

    α = rro / pw
    chunk.cgα[tt] = α
    rrn = ur!(chunk, settings.halodepth, α)

    β = rrn / rro
    chunk.cgβ[tt] = β
    p!(chunk, settings.halodepth, β)

    return rrn
end

# Initialises the CG solver
function init!(
    chunk::C,
    hd::Int,
    coefficient::Int,
    rx::Float64,
    ry::Float64,
) where {C<:Chunk}
    @assert coefficient in (CONDUCTIVITY, RECIP_CONDUCTIVITY)

    @. chunk.u = chunk.energy * chunk.density
    chunk.p .= 0.0
    chunk.r .= 0.0

    x, y = size(chunk)

    modifier = coefficient == CONDUCTIVITY ? 1 : -1
    H = halo(chunk, 1)
    @. chunk.w[H] = chunk.density[H]^modifier

    for jj = hd+1:y-1, kk = hd+1:x-1
        chunk.kx[kk, jj] =
            rx * (chunk.w[kk-1, jj] + chunk.w[kk, jj]) /
            (2chunk.w[kk-1, jj] * chunk.w[kk, jj])
        chunk.ky[kk, jj] =
            ry * (chunk.w[kk, jj-1] + chunk.w[kk, jj]) /
            (2chunk.w[kk, jj-1] * chunk.w[kk, jj])
    end

    H = halo(chunk, hd)
    chunk.w[H] .= smvp.(chunk, Ref(chunk.u), H)
    chunk.r[H] .= chunk.u[H] .- chunk.w[H]
    chunk.p[H] .= chunk.r[H]
    return sum(chunk.r[H] .* chunk.p[H])
end

# Calculates w
function w!(chunk::C, hd::Int) where {C<:Chunk}
    H = halo(chunk, hd)
    chunk.w[H] .= smvp.(chunk, Ref(chunk.p), H)
    return sum(chunk.w[H] .* chunk.p[H])
end

# Calculates u and r
function ur!(chunk::C, hd::Int, alpha::Float64) where {C<:Chunk}
    H = halo(chunk, hd)
    @. chunk.u[H] += alpha * chunk.p[H]
    @. chunk.r[H] -= alpha * chunk.w[H]
    return norm(chunk.r[H])
end

# Calculates p
function p!(chunk::C, hd::Int, beta::Float64) where {C<:Chunk}
    H = halo(chunk, hd)
    @. chunk.p[H] = beta * chunk.p[H] + chunk.r[H]
end

end
