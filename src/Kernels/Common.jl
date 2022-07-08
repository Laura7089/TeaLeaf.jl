module Common

using Base.Threads: @threads

export initkernel!
export kernelfinalise!
export blocksolve!
export blockinit!
export calcresidualkernel!
export calc2normkernel!
export diaginit!
export diagsolve!

CHUNK_LEFT = 1
CHUNK_RIGHT = 2
CHUNK_BOTTOM = 3
CHUNK_TOP = 4
EXTERNAL_FACE = -1

TL_PREC_NONE = 1
TL_PREC_JAC_DIAG = 2
TL_PREC_JAC_BLOCK = 3

CONDUCTIVITY = 1
RECIP_CONDUCTIVITY = 2

JAC_BLOCK_SIZE = 4
BLOCK_SIZE = 1
KSTEP = BLOCK_SIZE * JAC_BLOCK_SIZE

function initkernel!(
    xmin::Int,
    xmax::Int,
    ymin::Int,
    ymax::Int,
    haloxdepth::Int,
    zerobound::Vector{Bool},
    reflectbound::Bool,
    density::Matrix{Float64},
    energy::Matrix{Float64},
    u::Matrix{Float64},
    u0::Matrix{Float64},
    r::Matrix{Float64},
    w::Matrix{Float64},
    Kx::Matrix{Float64},
    Ky::Matrix{Float64},
    Di::Matrix{Float64},
    cp::Matrix{Float64},
    bfp::Matrix{Float64},
    Mi::Matrix{Float64},
    rx::Float64,
    ry::Float64,
    preconditioner_type::Int,
    coef::Int,
)
    @assert size(zerobound) == 4

    # TODO: solve index offsets that fortran's DIMENSION offers
    # TODO: convert to julia one-based indexing in for loops
    for param in [density, energy, u, r, w, Kx, Ky, Di, Mi, u0]
        @assert size(param) == (
            xmax + haloxdepth - (xmin - haloxdepth),
            ymax + haloxdepth - (ymin - haloxdepth),
        )
    end
    # REAL(KIND=8), DIMENSION(xmin:xmax,ymin:ymax) :: cp, bfp
    for param in [cp, bfp]
        @assert size(param) == (xmax - xmin, ymax - ymin)
    end

    u[:, :] = energy .* density
    u0[:, :] = energy .* density

    if coef == RECIP_CONDUCTIVITY
        w[:, :] = 1.0 ./ density
    elseif coef == CONDUCTIVITY
        w[:, :] = density
    end

    @threads for k = ymin-haloxdepth+1:ymax+haloxdepth
        for j = xmin-haloxdepth+1:xmax+haloxdepth
            Kx[j, k] = (w[j-1, k] + w[j, k]) / (2.0 * w[j-1, k] * w[j, k])
            Ky[j, k] = (w[j, k-1] + w[j, k]) / (2.0 * w[j, k-1] * w[j, k])
        end
    end

    # Whether to apply reflective boundary conditions to all external faces
    # TODO: fix zeros() size
    if !reflectbound
        if zerobound[CHUNK_LEFT]
            Kx[xmin-haloxdepth:xmin, :] = zeros()
        end
        if zerobound[CHUNK_RIGHT]
            Kx[xmax+1:xmax+haloxdepth, :] = zeros()
        end
        if zerobound[CHUNK_BUTTOM]
            Kx[:, ymin-haloxdepth:ymin] = zeros()
        end
        if zerobound[CHUNK_TOP]
            Kx[xmin-haloxdepth:xmax+haloxdepth, ymax+1:ymax+haloxdepth] = zeros()
        end
    end

    # Setup storage for the diagonal entries
    @threads for k = ymin-haloxdepth+1:ymax+haloxdepth-1
        for j = xmin-haloxdepth+1:xmax+haloxdepth-1
            Di[j, k] = (1.0 + ry * (Ky[j, k+1] + Ky[j, k]) + rx * (Kx[j+1, k] + Kx[j, k]))
        end
    end

    if preconditioner_type == TL_PREC_JAC_BLOCK
        blockinit(xmin, xmax, ymin, ymax, haloxdepth, cp, bfp, Kx, Ky, Di, rx, ry)
    elseif preconditioner_type == TL_PREC_JAC_DIAG
        diaginit(xmin, xmax, ymin, ymax, haloxdepth, Mi, Kx, Ky, Di, rx, ry)
    end

    @threads for k = ymin:ymax
        for j = xmin:xmax
            w[j, k] =
                Di[j, k] * u[j, k] - ry * (Ky[j, k+1] * u[j, k+1] + Ky[j, k] * u[j, k-1]) -
                rx * (Kx[j+1, k] * u[j+1, k] + Kx[j, k] * u[j-1, k])

            # r[j, k] = u[j, k] # This is required to make a zero initial guess to match petsc errant behaviour
            # Only works one timestep is run
        end
    end
    r[:, :] = u - w
end

function kernelfinalise!(
    xmin::Int,
    xmax::Int,
    ymin::Int,
    ymax::Int,
    haloxdepth::Int,
    energy::Matrix{Float64},
    density::Matrix{Float64},
    u::Matrix{Float64},
)
    for param in [u, energy, density]
        @assert size(param) = (
            xmax + haloxdepth - (xmin - haloxdepth),
            ymax + haloxdepth - (ymin - haloxdepth),
        )
    end

    # !$OMP PARALLEL
    # !$OMP DO
    #   DO k=ymin, ymax
    #     DO j=xmin, xmax
    #       energy(j,k) = u(j,k) / density(j,k)
    #     ENDDO
    #   ENDDO
    # !$OMP END DO
    # !$OMP END PARALLEL
    energy[:, :] = u ./ density
end

function calcresidualkernel!(
    xmin::Int,
    xmax::Int,
    ymin::Int,
    ymax::Int,
    haloxdepth::Int,
    u::Matrix{Float64},
    u0::Matrix{Float64},
    r::Matrix{Float64},
    Kx::Matrix{Float64},
    Ky::Matrix{Float64},
    Di::Matrix{Float64},
    rx::Float64,
    ry::Float64,
)
    for param in [Kx, u, r, Ky, u0, Di]
        @assert size(param) = (
            xmax + haloxdepth - (xmin - haloxdepth),
            ymax + haloxdepth - (ymin - haloxdepth),
        )
    end

    @threads for k = ymin:ymax
        for j = xmin:xmax
            smvp =
                Di[j, k] * u[j, k] - ry * (Ky[j, k+1] * u[j, k+1] + Ky[j, k] * u[j, k-1]) -
                rx * (Kx[j+1, k] * u[j+1, k] + Kx[j, k] * u[j-1, k])
            r(j, k) = u0[j, k] - smvp
        end
    end
end

# NOTE: you *must* use the result of this - it won't mutate the norm kernel
function calc2normkernel!(
    xmin::Int,
    xmax::Int,
    ymin::Int,
    ymax::Int,
    haloxdepth::Int,
    arr::Matrix{Float64},
    _::Float64,
)::Float64
    @assert size(arr) == (
        xmax + haloxdepth - (xmin - haloxdepth),
        ymax + haloxdepth - (ymin - haloxdepth),
    )

    # !$OMP PARALLEL
    # !$OMP DO REDUCTION(+:norm)
    #     DO k=ymin,ymax
    #         DO j=xmin,xmax
    #             norm = norm + arr(j, k)*arr(j, k)
    #         ENDDO
    #     ENDDO
    # !$OMP END DO
    # !$OMP END PARALLEL
    return sum(arr .^ 2)
end

function diaginit!(
    xmin::Int,
    xmax::Int,
    ymin::Int,
    ymax::Int,
    haloxdepth::Int,
    Mi::Matrix{Float64},
    Kx::Matrix{Float64},
    Ky::Matrix{Float64},
    Di::Matrix{Float64},
    rx::Float64,
    ry::Float64,
)
    for param in [Kx, Ky, Di, Mi]
        @assert size(param) == (
            xmax + haloxdepth - (xmin - haloxdepth),
            ymax + haloxdepth - (ymin - haloxdepth),
        )
    end

    omega = 1.0
    @threads for k = ymin-haloxdepth+1:ymax+haloxdepth-1
        for j = xmin-haloxdepth+1:xmax+haloxdepth-1
            if Di[j, k] != 0.0
                Mi[j, k] = omega / Di[j, k]
            else
                Mi[j, k] = 0.0_8
            end
        end
    end
end

function diagsolve!(
    xmin::Int,
    xmax::Int,
    ymin::Int,
    ymax::Int,
    haloxdepth::Int,
    depth::Int,
    r::Matrix{Float64},
    z::Matrix{Float64},
    Mi::Matrix{Float64},
)
    for param in [r, z, Mi]
        @assert size(param) == (
            xmax + haloxdepth - (xmin - haloxdepth),
            ymax + haloxdepth - (ymin - haloxdepth),
        )
    end

    # !$OMP DO
    #     DO k=ymin-depth,ymax+depth
    #       DO j=xmin-depth,xmax+depth
    #         z(j, k) = Mi(j, k)*r(j, k)
    #       ENDDO
    #     ENDDO
    # !$OMP END DO
    xr = xmin-depth:xmax+depth
    yr = ymin-depth:ymax+depth
    z[xr, yr] = Mi[xr, yr] .* r[xr, yr]
end

function blockinit!(
    xmin::Int,
    xmax::Int,
    ymin::Int,
    ymax::Int,
    haloxdepth::Int,
    cp::Matrix{Float64},
    bfp::Matrix{Float64},
    Kx::Matrix{Float64},
    Ky::Matrix{Float64},
    Di::Matrix{Float64},
    rx::Float64,
    ry::Float64,
)
    for param in [Kx, Ky, Di]
        @assert size(param) == (
            xmax + haloxdepth - (xmin - haloxdepth),
            ymax + haloxdepth - (ymin - haloxdepth),
        )
    end

    for param in [cp, bfp]
        @assert size(param) == (xmax - xmin, ymax - ymin)
    end

    @threads for ko = ymin:JAC_BLOCK_SIZE:ymax
        bot = ko
        top = min(ko + JAC_BLOCK_SIZE - 1, ymax)

        for j = xmin:xmax
            k = bot
            cp[j, k] = (-Ky[j, k+1] * ry) / Di[j, k]

            for k = bot+1:top
                bfp[j, k] = 1.0 / (Di[j, k] - (-Ky[j, k] * ry) * cp[j, k-1])
                cp[j, k] = (-Ky[j, k+1] * ry) * bfp[j, k]
            end
        end
    end
end

function blocksolve!(
    xmin::Int,
    xmax::Int,
    ymin::Int,
    ymax::Int,
    haloxdepth::Int,
    r::Matrix{Float64},
    z::Matrix{Float64},
    cp::Matrix{Float64},
    bfp::Matrix{Float64},
    Kx::Matrix{Float64},
    Ky::Matrix{Float64},
    Di::Matrix{Float64},
    rx::Float64,
    ry::Float64,
)
    for param in [Kx, Ky, Di, r, z]
        @assert size(param) == (
            xmax + haloxdepth - (xmin - haloxdepth),
            ymax + haloxdepth - (ymin - haloxdepth),
        )
    end

    for param in [cp, bfp]
        @assert size(param) == (xmax - xmin, ymax - ymin)
    end

    dp_l = zeros(JAC_BLOCK_SIZE - 1)
    z_l = zeros(JAC_BLOCK_SIZE - 1)

    k_extra = ymax - (ymax % KSTEP)

    @threads for ko = ymin:KSTEP:k_extra
        upper_k = ko + KSTEP - JAC_BLOCK_SIZE

        for ki = ko:JAC_BLOCK_SIZE:upper_k
            bot = ki
            top = ki + JAC_BLOCK_SIZE - 1

            for j = xmin:xmax
                dp_l[1] = r[j, k] / Di[j, k]

                # for k = bot+1:top
                #     dp_l[k-bot] =
                #         (r[j, k] - (-Ky[j, k] * ry) * dp_l[k-bot-1]) * bfp[j, k]
                # end
                _r = bot+2:top+1
                dp_l[2:top-bot+1] .=
                    (r[j, _r] - (Ky[j, _r] .* -ry) .* dp_l[1:top-bot] .* bfp[j, _r])

                z_l[top-bot] = dp_l[k-bot]

                # for k = top-1:-1:bot
                #     z_l[k-bot] = dp_l[k-bot] - cp[j, k] * z_l[k-bot+1]
                # end
                z_l[1:top-bot] .= dp_l[1:top-bot] - cp[j, bot+1:top] .* z_l[2:top-bot+1]

                # for k = bot:top
                #     z[j, k] = z_l[k-bot]
                # end
                z[j, bot+1:top+1] .= z_l[1:top-bot+1]
            end
        end

        @threads for ki = k_extra+1:JAC_BLOCK_SIZE:ymax
            bot = min(ki, ymax)
            top = min(ki + JAC_BLOCK_SIZE - 1, ymax)

            for j = xmin:xmax
                dp_l[1] = r[j, k] / Di[j, k]

                for k = bot+1:top
                    dp_l[k-bot] = (r[j, k] - (-Ky[j, k] * ry) * dp_l[k-bot-1]) * bfp[j, k]
                end

                z_l[top-bot] = dp_l[k-bot]

                # for k in top-1:-1:bot
                #     z_l[k-bot] = dp_l[k-bot] - cp[j, k] * z_l[k-bot+1]
                # end
                z_l[1:top-bot] .= dp_l[1:top-bot] - cp[j, bot:top] .* z_l[2:top-bot+1]

                z[j, bot:top] .= z_l[1:top-bot]
            end
        end
    end
end

end
