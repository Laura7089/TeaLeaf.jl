module CG

import TeaLeaf.Chunk
import TeaLeaf.Settings
import TeaLeaf.CONDUCTIVITY
import TeaLeaf.smvp

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

    # Iterate till convergence
    for tt = 1:settings.max_iters
        rro, error = main_step(chunk, settings, tt, rro) # Done

        halo_update!(chunk, settings, 1) # Done

        if sqrt(abs(error)) < settings.eps
            break
        end
    end

    @info "Iterations" tt
    return error
end

# Invokes the CG initialisation kernels
function init_driver!(
    chunk::C,
    set::Settings,
    rx::Float64,
    ry::Float64,
) where {C<:Chunk}
    rro = init!(chunk, set.halo_depth, set.coefficient, rx, ry)

    # Need to update for the matvec
    reset_fields_to_exchange(settings) # Done
    set.fields_to_exchange[FIELD_U] = true
    set.fields_to_exchange[FIELD_P] = true
    halo_update!(chunk, settings, 1) # Done

    copy_u(chunk, set.halo_depth) # Done

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
        index = kk + (jj - 1) * chunk.x
        chunk.u[index] = chunk.energy[index] * chunk.density[index]
    end
    chunk.p .= 0.0
    chunk.r .= 0.0

    for jj = 2:chunk.y-1, kk = 2:chunk.x-1
        index = kk + jj * chunk.x
        chunk.w[index] =
            (coefficient == CONDUCTIVITY) ? chunk.density[index] :
            1.0 / chunk.density[index]
    end

    for jj = hd+1:chunk.y-1, kk = hd+1:chunk.x-1
        index = kk + jj * chunk.x
        chunk.kx[index] =
            rx * (chunk.w[index-1] + chunk.w[index]) /
            (2.0 * chunk.w[index-1] * chunk.w[index])
        chunk.ky[index] =
            ry * (chunk.w[index-chunk.x] + chunk.w[index]) /
            (2.0 * chunk.w[index-chunk.x] * chunk.w[index])
    end

    rro_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
        p = smvp(chunk, chunk.u, index)
        chunk.w[index] = p
        chunk.r[index] = chunk.u[index] - chunk.w[index]
        chunk.p[index] = chunk.r[index]
        rro_temp += chunk.r[index] * chunk.p[index]
    end

    return rro_temp
end

# Calculates w
function calc_w(chunk::C, hd::Int) where {C<:Chunk}
    pw_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
        p = smvp(chunk, chunk.p, index)
        chunk.w[index] = p
        pw_temp += chunk.w[index] * chunk.p[index]
    end

    return pw_temp
end

# Calculates u and r
function calc_ur(chunk::C, hd::Int, alpha::Float64) where {C<:Chunk}
    rrn_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x

        chunk.u[index] += alpha * chunk.p[index]
        chunk.r[index] -= alpha * chunk.w[index]
        rrn_temp += chunk.r[index] * chunk.r[index]
    end

    return rrn_temp
end

# Calculates p
function calc_p(chunk::C, hd::Int, beta::Float64) where {C<:Chunk}
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x

        chunk.p[index] = beta * chunk.p[index] + chunk.r[index]
    end
end

end
