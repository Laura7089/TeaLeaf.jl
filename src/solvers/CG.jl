module CG

import TeaLeaf.Chunk
import TeaLeaf.Settings
import TeaLeaf.CONDUCTIVITY
import TeaLeaf.smvp
import TeaLeaf.FIELD_U
import TeaLeaf.FIELD_P
import TeaLeaf.halo_update!
import TeaLeaf.copy_u!
import TeaLeaf.ERROR_START

# Performs a full solve with the CG solver kernels
function driver!(
    chunk::C,
    settings::Settings,
    rx::Float64,
    ry::Float64,
)::Float64 where {C<:Chunk} # TODO: modifies `error`
    # Perform CG initialisation
    rro = init_driver!(chunk, settings, rx, ry) # Done

    error = ERROR_START

    iters = 0
    # Iterate till convergence
    for tt = 1:settings.max_iters
        iters += 1
        rro, error = main_step(chunk, settings, tt, rro) # Done

        halo_update!(chunk, settings, 1) # Done

        if sqrt(abs(error)) < settings.eps
            break
        end
    end

    @info "Iterations" iters
    return error
end

# Invokes the CG initialisation kernels
function init_driver!(chunk::C, set::Settings, rx::Float64, ry::Float64) where {C<:Chunk}
    rro = init!(chunk, set.halo_depth, set.coefficient, rx, ry)

    # Need to update for the matvec
    set.fields_to_exchange .= false
    set.fields_to_exchange[FIELD_U] = true
    set.fields_to_exchange[FIELD_P] = true
    halo_update!(chunk, set, 1)

    copy_u!(chunk, set.halo_depth)

    return rro
end

# Invokes the main CG solve kernels
function main_step(
    chunk::C,
    settings::Settings,
    tt::Int,
    rro::Float64,
)::Tuple{Float64,Float64} where {C<:Chunk}
    pw = calc_w(chunk, settings.halo_depth) |> sum

    α = rro / pw

    # TODO: Some redundancy across chunks??
    chunk.cg_alphas[tt] = α

    rrn = calc_ur(chunk, settings.halo_depth, α) # Done

    β = rrn / rro

    # TODO: Some redundancy across chunks??
    chunk.cg_betas[tt] = β

    calc_p(chunk, settings.halo_depth, β) # Done

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
function calc_w(chunk::C, hd::Int) where {C<:Chunk}
    pw_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        p = smvp(chunk, chunk.p, kk, jj)
        chunk.w[kk, jj] = p
        pw_temp += chunk.w[kk, jj] * chunk.p[kk, jj]
    end

    return pw_temp
end

# Calculates u and r
function calc_ur(chunk::C, hd::Int, alpha::Float64) where {C<:Chunk}
    rrn_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        chunk.u[kk, jj] += alpha * chunk.p[kk, jj]
        chunk.r[kk, jj] -= alpha * chunk.w[kk, jj]
        rrn_temp += chunk.r[kk, jj] * chunk.r[kk, jj]
    end

    return rrn_temp
end

# Calculates p
function calc_p(chunk::C, hd::Int, beta::Float64) where {C<:Chunk}
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        chunk.p[kk, jj] = beta * chunk.p[kk, jj] + chunk.r[kk, jj]
    end
end

end
