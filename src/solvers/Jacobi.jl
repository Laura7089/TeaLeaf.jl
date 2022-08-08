module Jacobi

import ..Chunk
import ..Settings

# Performs a full solve with the Jacobi solver kernels
function driver(
    chunks::Vector{Chunk},
    settings::Settings,
    rx::Float64,
    ry::Float64,
    error::Float64,
)::Float64
    init_driver(chunks, settings, rx, ry) # Done
    final_time = 0

    # Iterate till convergence
    for tt = 1:settings.max_iters
        error += main_step_driver(chunks, settings, tt) # Done
        @debug "" error

        halo_update!(chunks, settings, 1)

        final_time = tt
        if abs(error) < settings.eps
            break
        end
    end

    @info "Jacobi" final_time
    return error
end

# Invokes the CG initialisation kernels
function init_driver(chunks::Vector{Chunk}, settings::Settings, rx::Float64, ry::Float64)
    for c in chunks
        init(c, settings.halo_depth, settings.coefficient, rx, ry)
        copy_u(c, settings.halo_depth)
    end

    # Need to update for the matvec
    settings.fields_to_exchange .= false
    settings.fields_to_exchange[FIELD_U] = true
end

# Invokes the main Jacobi solve kernels
function main_step_driver(chunks::Vector{Chunk}, settings::Settings, tt::Int)::Float64 # TODO: returns error
    error = iterate.(chunks, settings.halo_depth) |> sum

    if tt % 50 == 0
        halo_update!(chunks, settings, 1)

        for c in chunks
            calculate_residual(c, settings.halo_depth)
            # TODO: how does this mutate `error`?
            error += calculate_2norm(c, settings.halo_depth, c.r)
        end
    end

    return error
end

# Initialises the Jacobi solver
function init(chunk::Chunk, hd::Int, coef::Int, rx::Float64, ry::Float64)
    if coef < CONDUCTIVITY && coef < RECIP_CONDUCTIVITY
        throw("Coefficient $(coef) is not valid.")
    end

    index = @. (1:chunk.x-1) + (1:chunk.y-1) * chunk.x
    temp = chunk.energy[index] .* chunk.density[index]
    chunk.u0[index] .= temp
    chunk.u[index] .= temp

    for jj = hd+1:chunk.y-1, kk = hd+1:chunk.x-1
        index = kk + jj * chunk.x
        densc = (coef == CONDUCTIVITY) ? chunk.density[index] : 1.0 / chunk.density[index]
        densl =
            (coef == CONDUCTIVITY) ? chunk.density[index-1] : 1.0 / chunk.density[index-1]
        densd =
            (coef == CONDUCTIVITY) ? chunk.density[index-chunk.x] :
            1.0 / chunk.density[index-chunk.x]

        chunk.kx[index] = rx * (densl + densc) / (2.0 * densl * densc)
        chunk.ky[index] = ry * (densd + densc) / (2.0 * densd * densc)
    end
end

# The main Jacobi solve step
function iterate(chunk::Chunk, hd::Int)::Float64
    index = @. (1:chunk.x) + ((0:chunk.y-1) * chunk.x)
    chunk.r[index] .= chunk.u[index]

    err = 0.0
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + (jj - 1) * chunk.x
        chunk.u[index] =
            (
                chunk.u0[index] +
                (
                    chunk.kx[index+1] * chunk.r[index+1] +
                    chunk.kx[index] * chunk.r[index-1]
                ) +
                (
                    chunk.ky[index+chunk.x] * chunk.r[index+chunk.x] +
                    chunk.ky[index] * chunk.r[index-chunk.x]
                )
            ) / (
                1.0 +
                (chunk.kx[index] + chunk.kx[index+1]) +
                (chunk.ky[index] + chunk.ky[index+chunk.x])
            )

        err += abs(chunk.u[index] - chunk.r[index])
    end

    return err
end

end
