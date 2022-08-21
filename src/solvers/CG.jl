module CG

using TeaLeaf
using TeaLeaf.Kernels

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
function mainstep!(chunk::C, set::Settings, tt::Int, rro::Float64)::Float64 where {C<:Chunk}
    pw = w!(chunk, set.halodepth)

    α = rro / pw
    chunk.cgα[tt] = α
    rrn = ur!(chunk, set.halodepth, α)

    β = rrn / rro
    chunk.cgβ[tt] = β
    p!(chunk, set.halodepth, β)

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

    modifier = coefficient == CONDUCTIVITY ? 1 : -1
    H = halo(chunk, 1) # Note hardcoded 1
    @. chunk.w[H] = chunk.density[H]^modifier

    x, y = size(chunk)
    for jj = hd+1:y-1, kk = hd+1:x-1
        chunk.kx[kk, jj] =
            rx * (chunk.w[kk-1, jj] + chunk.w[kk, jj]) /
            (2chunk.w[kk-1, jj] * chunk.w[kk, jj])
        chunk.ky[kk, jj] =
            ry * (chunk.w[kk, jj-1] + chunk.w[kk, jj]) /
            (2chunk.w[kk, jj-1] * chunk.w[kk, jj])
    end

    temp = 0
    xs, ys = haloa(chunk, hd)
    for jj in ys, kk in xs
        chunk.w[kk, jj] = smvp(chunk, chunk.u, CartesianIndex(kk, jj))
        chunk.r[kk, jj] = chunk.u[kk, jj] - chunk.w[kk, jj]
        chunk.p[kk, jj] = chunk.r[kk, jj]
        temp += chunk.r[kk, jj]^2
    end
    return temp
end

# Calculates w
function w!(chunk::C, hd::Int) where {C<:Chunk}
    temp = 0
    xs, ys = haloa(chunk, hd)
    for jj in ys, kk in xs
        chunk.w[kk, jj] = smvp(chunk, chunk.p, CartesianIndex(kk, jj))
        temp += chunk.w[kk, jj] * chunk.p[kk, jj]
    end
    return temp
end

# Calculates u and r
function ur!(chunk::C, hd::Int, α::Float64) where {C<:Chunk}
    H = halo(chunk, hd)
    @. chunk.u[H] += α * chunk.p[H]
    @. chunk.r[H] -= α * chunk.w[H]
    return sum(x -> x^2, chunk.r)
end

# Calculates p
function p!(chunk::C, hd::Int, β::Float64) where {C<:Chunk}
    H = halo(chunk, hd)
    @. chunk.p[H] = β * chunk.p[H] + chunk.r[H]
end

end
