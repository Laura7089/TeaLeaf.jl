module CG

using TeaLeaf
using TeaLeaf.Kernels

# Performs a full solve with the CG solver kernels
function driver!(
    chunk::C,
    set::Settings,
    rx::Float64,
    ry::Float64,
)::Float64 where {C<:Chunk}
    # Perform CG initialisation
    rro = init!(chunk, set.halodepth, set.coefficient, rx, ry)

    setindex!.(Ref(set.toexchange), false, CHUNK_FIELDS)
    set.toexchange[:u] = true
    set.toexchange[:p] = true
    haloupdate!(chunk, set, 1)

    copyu!(chunk, set.halodepth)

    error = ERROR_START

    iters = 0
    # Iterate till convergence
    for tt = 1:set.maxiters
        iters = tt
        rro, error = mainstep(chunk, set, tt, rro)

        haloupdate!(chunk, set, 1)

        if sqrt(abs(error)) < set.eps
            break
        end
    end

    @info "Iterations" iters
    return error
end

# Invokes the main CG solve kernels
function mainstep(
    chunk::C,
    settings::Settings,
    tt::Int,
    rro::Float64,
)::Tuple{Float64,Float64} where {C<:Chunk}
    pw = w!(chunk, settings.halodepth)

    α = rro / pw
    chunk.cgα[tt] = α
    rrn = ur!(chunk, settings.halodepth, α)

    β = rrn / rro
    chunk.cgβ[tt] = β
    p!(chunk, settings.halodepth, β)

    return (rrn, rrn)
end

# Initialises the CG solver
function init!(
    chunk::C,
    hd::Int,
    coefficient::Int,
    rx::Float64,
    ry::Float64,
) where {C<:Chunk}
    if coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY
        throw("Coefficient $(coefficient) is not valid.")
    end

    for jj = 1:chunk.y, kk = 1:chunk.x
        chunk.u[kk, jj] = chunk.energy[kk, jj] * chunk.density[kk, jj]
    end
    chunk.p .= 0.0
    chunk.r .= 0.0

    for jj = 2:chunk.y-1, kk = 2:chunk.x-1
        chunk.w[kk, jj] =
            (coefficient == CONDUCTIVITY) ? chunk.density[kk, jj] :
            1.0 / chunk.density[kk, jj]
    end

    for jj = hd+1:chunk.y-1, kk = hd+1:chunk.x-1
        chunk.kx[kk, jj] =
            rx * (chunk.w[kk-1, jj] + chunk.w[kk, jj]) /
            (2.0 * chunk.w[kk-1, jj] * chunk.w[kk, jj])
        chunk.ky[kk, jj] =
            ry * (chunk.w[kk, jj-1] + chunk.w[kk, jj]) /
            (2.0 * chunk.w[kk, jj-1] * chunk.w[kk, jj])
    end

    rro_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        p = smvp(chunk, chunk.u, kk, jj)
        chunk.w[kk, jj] = p
        chunk.r[kk, jj] = chunk.u[kk, jj] - chunk.w[kk, jj]
        chunk.p[kk, jj] = chunk.r[kk, jj]
        rro_temp += chunk.r[kk, jj] * chunk.p[kk, jj]
    end

    return rro_temp
end

# Calculates w
function w!(chunk::C, hd::Int) where {C<:Chunk}
    pw_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        p = smvp(chunk, chunk.p, kk, jj)
        chunk.w[kk, jj] = p
        pw_temp += chunk.w[kk, jj] * chunk.p[kk, jj]
    end

    return pw_temp
end

# Calculates u and r
function ur!(chunk::C, hd::Int, alpha::Float64) where {C<:Chunk}
    rrn_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        chunk.u[kk, jj] += alpha * chunk.p[kk, jj]
        chunk.r[kk, jj] -= alpha * chunk.w[kk, jj]
        rrn_temp += chunk.r[kk, jj] * chunk.r[kk, jj]
    end

    return rrn_temp
end

# Calculates p
function p!(chunk::C, hd::Int, beta::Float64) where {C<:Chunk}
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        chunk.p[kk, jj] = beta * chunk.p[kk, jj] + chunk.r[kk, jj]
    end
end

end