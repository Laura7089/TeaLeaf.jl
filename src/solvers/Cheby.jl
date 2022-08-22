module Cheby

using TeaLeaf
using TeaLeaf.Kernels
import ..CG

# Performs full solve with the Chebyshev kernels
function solve!(chunk::C, settings::Settings, rx::Float64, ry::Float64) where {C<:Chunk}
    error = ERROR_START
    rro = 0
    estiter = 0
    chebyiters = 0

    # Perform CG initialisation
    rro = CG.init!(chunk, settings.halodepth, settings.coefficient, rx, ry)

    # Iterate till convergence
    for tt = 1:settings.maxiters
        # If we have already ran cheby iterations, continue
        # If we are error switching, check the error
        # If not error switching, perform preset iterations
        # Perform enough iterations to converge eigenvalues
        switchcheby =
            chebyiters != 0 || (
                settings.errorswitch ?
                (error < settings.epslim) && (tt > CG_ITERS_FOR_EIGENVALUES) :
                (tt > settings.presteps) && (error < ERROR_SWITCH_MAX)
            )

        if !switchcheby
            # Perform a CG iteration
            error = CG.mainstep!(chunk, settings, tt, rro)
        else
            chebyiters += 1

            # Check if first step
            if chebyiters == 1
                # Initialise the solver
                bb = init!(chunk, settings, tt)

                # Perform the main step
                error = mainstep!(chunk, settings, chebyiters, true, error)

                # Estimate the number of Chebyshev iterations
                estiter = calciter(chunk, error, bb)
            else
                calc2norm = (chebyiters >= estiter) && ((tt + 1) % 10 == 0)

                # Perform main step
                error = mainstep!(chunk, settings, chebyiters, calc2norm, error)
            end
        end

        haloupdate!(chunk, settings, 1)

        if abs(error) < settings.eps
            break
        end
    end

    @info "Cheby solve complete" chebyiters estiter
end

# Invokes the Chebyshev initialisation kernels
function init!(chunk::C, settings::Settings, cgiters::Int)::Float64 where {C<:Chunk}
    # Initialise eigenvalues and Chebyshev coefficients
    eigenvalues!(chunk, settings, cgiters)
    coef!(chunk, settings, settings.maxiters - cgiters)
    bb = sum(x -> x^2, chunk.u0[halo(chunk, settings.halodepth)])

    xs, ys = haloa(chunk, settings.halodepth)
    for jj in ys, kk in xs
        chunk.w[kk, jj] = smvp(chunk, chunk.u, CartesianIndex(kk, jj))
        chunk.r[kk, jj] = chunk.u0[kk, jj] - chunk.w[kk, jj]
        chunk.p[kk, jj] = chunk.r[kk, jj] / chunk.θ
    end
    chunk.u[xs, ys] .+= chunk.p[xs, ys]

    resettoexchange!(settings)
    settings.toexchange[:u] = true
    haloupdate!(chunk, settings, 1)

    return bb
end

# Performs the main iteration step
function mainstep!(
    chunk::C,
    settings::Settings,
    chebyiters::Int,
    calc2norm::Bool,
    error::Float64,
)::Float64 where {C<:Chunk}
    xs, ys = haloa(chunk, settings.halodepth)
    for jj in ys, kk in xs
        chunk.w[kk, jj] = smvp(chunk, chunk.u, CartesianIndex(kk, jj))
        chunk.r[kk, jj] = chunk.u0[kk, jj] - chunk.w[kk, jj]
        chunk.p[kk, jj] =
            chunk.chebyα[chebyiters+1] * chunk.p[kk, jj] +
            chunk.chebyβ[chebyiters+1] * chunk.r[kk, jj]
    end
    H = halo(chunk, settings.halodepth)
    chunk.u[H] .+= chunk.p[H]

    if calc2norm
        error = sum(x -> x^2, chunk.r[halo(chunk, settings.halodepth)])
    end
    return error
end

# Calculates the estimated iterations for Chebyshev solver
function calciter(chunk::C, error::Float64, bb::Float64)::Int where {C<:Chunk}
    connum = chunk.eigmax / chunk.eigmin

    # Calculate estimated iteration count
    itα = eps(Float64) * bb / 4error
    γ = (sqrt(connum) - 1) / (sqrt(connum) + 1)

    # TODO is log base 10 correct?
    return round(log(10, itα) / 2log(10, γ))
end

# Calculates the Chebyshev coefficients for the chunk
function coef!(chunk::C, settings::Settings, maxiters::Int) where {C<:Chunk}
    chunk.θ = (chunk.eigmax + chunk.eigmin) / 2
    δ = (chunk.eigmax - chunk.eigmin) / 2
    σ = chunk.θ / δ
    ρₒ = 1 / σ

    for ii = 1:maxiters
        ρₙ = 1 / (2σ - ρₒ)
        α = ρₙ * ρₒ
        β = 2ρₙ / δ
        chunk.chebyα[ii] = α
        chunk.chebyβ[ii] = β
        ρₒ = ρₙ
    end
end

end
