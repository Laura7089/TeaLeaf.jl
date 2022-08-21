module Jacobi

using TeaLeaf
using TeaLeaf.Kernels

# Performs a full solve with the Jacobi solver kernels
function driver!(
    chunk::C,
    settings::Settings,
    rx::Float64,
    ry::Float64,
)::Float64 where {C<:Chunk}
    init!(chunk, settings, rx, ry)
    final_time = 0
    error = ERROR_START

    # Iterate till convergence
    for tt = 1:settings.maxiters
        error += iterate!(chunk, settings.halodepth)

        if tt % 50 == 0
            haloupdate!(chunk, settings, 1)

            residual!(chunk, settings.halodepth)
            error += sum(x -> x^2, (chunk.r[halo(chunk, settings.halodepth)]))
        end

        haloupdate!(chunk, settings, 1)

        final_time = tt
        if abs(error) < settings.eps
            break
        end
    end

    @info "Jacobi" final_time
    return error
end

function init!(chunk::C, set::Settings, rx::Float64, ry::Float64) where {C<:Chunk}
    if set.coefficient < min(CONDUCTIVITY, RECIP_CONDUCTIVITY)
        throw("Coefficient $(set.coefficient) is not valid.")
    end

    x, y = size(chunk)
    temp = chunk.energy .* chunk.density
    chunk.u0 .= temp
    chunk.u .= temp

    p = set.coefficient == CONDUCTIVITY ? 1 : -1
    for jj = set.halodepth+1:y-1, kk = set.halodepth+1:x-1
        densc = chunk.density[kk, jj]^p
        densl = chunk.density[kk-1, jj]^p
        densd = chunk.density[kk, jj-1]^p

        chunk.kx[kk, jj] = rx * (densl + densc) / (2densl * densc)
        chunk.ky[kk, jj] = ry * (densd + densc) / (2densd * densc)
    end

    copyu!(chunk, set.halodepth)

    resettoexchange!(set)
    set.toexchange[:u] = true
end

function iterate!(chunk::Chunk, hd::Int)::Float64
    x, y = size(chunk)
    chunk.r .= chunk.u
    for jj = hd+1:y-hd, kk = hd+1:x-hd
        chunk.u[kk, jj] =
            (
                chunk.u0[kk, jj] +
                chunk.kx[kk+1, jj] * chunk.r[kk+1, jj] +
                chunk.kx[kk, jj] * chunk.r[kk-1, jj] +
                chunk.ky[kk, jj+1] * chunk.r[kk, jj+1] +
                chunk.ky[kk, jj] * chunk.r[kk, jj-1]
            ) / (
                1 +
                chunk.kx[kk, jj] +
                chunk.kx[kk+1, jj] +
                chunk.ky[kk, jj] +
                chunk.ky[kk, jj+1]
            )
    end

    return abs.(chunk.u .- chunk.r) |> sum
end

end
