module PPCG

using TeaLeaf
import ..Chunk
import ..Settings
import ..CG
import ..Cheby

# Performs a full solve with the PPCG solver
function solve!(
    chunk::C,
    settings::Settings,
    rx::Float64,
    ry::Float64,
    error::Float64,
) where {C<:Chunk}
    ppcgiters = 0

    # Perform CG initialisation
    rro = CG.init!(chunk, settings.halodepth, settings.coefficient, rx, ry)

    # Iterate till convergence
    for tt = 1:settings.max_iters
        # If we have already ran PPCG inner iterations, continue
        # If we are error switching, check the error
        # If not error switching, perform preset iterations
        # Perform enough iterations to converge eigenvalues
        switchppcg =
            ppcgiters != 0 || (
                settings.error_switch ?
                (error < settings.eps_lim) && (tt > CG_ITERS_FOR_EIGENVALUES) :
                (tt > settings.presteps) && (error < ERROR_SWITCH_MAX)
            )

        if !switchppcg
            # Perform a CG iteration
            rro = error = CG.mainstep!(chunk, settings, tt, rro)
        else
            ppcgiters ++

            # If first step perform initialisation
            if ppcgiters == 1
                # Initialise the eigenvalues and Chebyshev coefficients
                eigenvalues!(chunk, settings, tt)
                Cheby.coef!(chunk, settings, settings.ppcginnersteps)

                init!(chunk, settings)
            end
            rro = error = mainstep!(chunk, settings, rro)
        end

        haloupdate!(chunk, settings.halodepth, 1)

        if abs(error) < settings.eps
            break
        end
    end

    @info "PPCG solve complete" iters error
end

# Invokes the PPCG initialisation kernels
function init!(chunk::C, settings::Settings) where {C<:Chunk}
    residual!(chunk, settings.halodepth)

    resettoexhange!(settings)
    settings.toexchange[:p] = true
    haloupdate!(chunk, settings.hd, 1)
end

# Invokes the main PPCG solver kernels
function mainstep!(
    chunk::C,
    settings::Settings,
    rro::Float64,
    error::Float64,
)::Float64 where {C<:Chunk}
    pw = CG.w!(chunk, settings.halodepth)
    α = rro / pw
    rrn = CG.ur!(chunk, settings.halodepth, α)

    # Perform the inner iterations
    init!(chunk, settings)

    resettoexchange!(settings)
    settings.toexchange[:sd] = true

    for pp = 1:settings.ppcginnersteps
        haloupdate!(chunk, settings, 1)
        xs, ys = haloa(chunk, settings.halodepth)
        for jj in ys, kk in xs
            chunk.r[kk, jj] -= smvp(chunk, chunk.sd, CartesianIndex(kk, jj))
            chunk.u[kk, jj] += chunk.sd[kk, jj]
            chunk.sd[kk, jj] =
                chunk.chebyα[pp] * chunk.sd[kk, jj] + chunk.chebyβ[pp] * chunk.r[kk, jj]
        end
    end

    resettoexchange!(settings)
    settings.toexchange[:p] = true
    rrn = sum(x -> x^2, chunk.r[halo(chunk, setting)])

    β = rrn / rro

    CG.p!(chunk, settings.halodepth, β)
    return rrn
end

# Initialises the PPCG solver
function init!(chunk::Chunk, hd::Int)
    H = halo(chunk, hd)
    @. chunk.sd[H] = chunk.r[H] / chunk.θ
end

end
