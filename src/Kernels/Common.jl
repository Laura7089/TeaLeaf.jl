module Common

import Threads: threads

export tea_leaf_common_init_kernel!
export tea_leaf_kernel_finalise!
export tea_block_solve!
export tea_block_init!
export tea_leaf_calc_residual_kernel!
export tea_leaf_calc_2norm_kernel!
export tea_diag_init!
export tea_diag_solve!

CHUNK_LEFT=1
CHUNK_RIGHT=2
CHUNK_BOTTOM=3
CHUNK_TOP=4
EXTERNAL_FACE=-1

TL_PREC_NONE= 1
TL_PREC_JAC_DIAG= 2
TL_PREC_JAC_BLOCK= 3

CONDUCTIVITY= 1
RECIP_CONDUCTIVITY= 2

jac_block_size= 4
block_size=1
kstep= block_size*jac_block_size

function tea_leaf_common_init_kernel!(x_min::Int, x_max::Int, y_min::Int, y_max::Int, halo_exchange_depth::Int, zero_boundary::Vector{Bool}, reflective_boundary::Bool, density::Matrix{Float64}, energy::Matrix{Float64}, u::Matrix{Float64}, u0::Matrix{Float64}, r::Matrix{Float64}, w::Matrix{Float64}, Kx::Matrix{Float64}, Ky::Matrix{Float64}, Di::Matrix{Float64}, cp::Matrix{Float64}, bfp::Matrix{Float64}, Mi::Matrix{Float64}, rx::Float64, ry::Float64, preconditioner_type::Int, coef::Int)
    @assert size(zero_boundary) == 4

    # TODO: solve index offsets that fortran's DIMENSION offers
    # REAL(KIND=8), DIMENSION(x_min-halo_exchange_depth:x_max+halo_exchange_depth,y_min-halo_exchange_depth:y_max+halo_exchange_depth) &
    for param in [density, energy, u, r, w, Kx, Ky, Di, Mi, u0]
        @assert size(param) == (x_max+halo_exchange_depth - (x_min-halo_exchange_depth),y_max+halo_exchange_depth- (y_min-halo_exchange_depth))
    end
    # REAL(KIND=8), DIMENSION(x_min:x_max,y_min:y_max) :: cp, bfp
    for param in [cp, bfp]
        @assert size(param) == (x_max - x_min, y_max - y_min)
    end

    u[:,:] = energy .* density
    u0[:,:] = energy .* density

    if coef == RECIP_CONDUCTIVITY
        # use w as temp val
        # @threads for k in (y_min-halo_exchange_depth):(y_max+halo_exchange_depth)
        #     for j in x_min-halo_exchange_depth:(x_max+halo_exchange_depth)
        #         w[j,k]=1.0/density[j  ,k  ]
        #     end
        # end
        w[:,:] = 1.0 ./ density
    else if coef == CONDUCTIVITY
        # @threads for k in (y_min-halo_exchange_depth):(y_max+halo_exchange_depth)
        #     for j in (x_min-halo_exchange_depth):(x_max+halo_exchange_depth)
        #         w[j  ,k  ]=density[j  ,k  ]
        #     end
        # end
        w[:,:] = density
    end

    @threads for k in (y_min-halo_exchange_depth + 1):(y_max+halo_exchange_depth)
        for j in (x_min-halo_exchange_depth + 1):(x_max+halo_exchange_depth)
            Kx[j,k]=(w[j-1,k  ] + w[j,k])/(2.0*w[j-1,k  ]*w[j,k])
            Ky[j,k]=(w[j  ,k-1] + w[j,k])/(2.0*w[j  ,k-1]*w[j,k])
        end
    end

    # Whether to apply reflective boundary conditions to all external faces
    if !reflective_boundary
        if zero_boundary[CHUNK_LEFT]
            # @threads for k in (y_min-halo_exchange_depth):(y_max+halo_exchange_depth)
            #     for j in (x_min-halo_exchange_depth):x_min
            #         Kx[j,k]=0.0
            #     end
            # end
            Kx[(x_min - halo_exchange_depth):x_min,:] = zeros()
        end

        if zero_boundary[CHUNK_RIGHT]
            # @threads for  k in (y_min-halo_exchange_depth):(y_max+halo_exchange_depth)
            #     for j in (x_max + 1):(x_max+halo_exchange_depth)
            #         Kx[j,k]=0.0
            #     end
            # end
            Kx[(x_max + 1):(x_max + halo_exchange_depth),:] = zeros()
        en

        if zero_boundary[CHUNK_BUTTOM]
            # @threads for k in (y_min-halo_exchange_depth):y_min
            #     for j in (x_min-halo_exchange_depth):(x_max+halo_exchange_depth)
            #         Ky[j,k]=0.0
            #     end
            # end
            Kx[:,(y_min-halo_exchange_depth):y_min] = zeros()
        end

        if zero_boundary[CHUNK_TOP]
            # @threads for k in (y_max + 1):(y_max+halo_exchange_depth)
            #     for j in (x_min-halo_exchange_depth):(x_max+halo_exchange_depth)
            #         Ky[j,k]=0.0
            #     end
            # end
            Kx[(x_min-halo_exchange_depth):(x_max+halo_exchange_depth),(y_max + 1):(y_max+halo_exchange_depth)] = zeros()
        end
    end

    # Setup storage for the diagonal entries
    @threads for k in (y_min-halo_exchange_depth+1):(y_max+halo_exchange_depth-1)
        for j in (x_min-halo_exchange_depth+1):(x_max+halo_exchange_depth-1)
            Di[j,k]=(1.0+ ry*(Ky[j, k+1] + Ky[j, k])+ rx*(Kx[j+1, k] + Kx[j, k]))
        end
    end

    if preconditioner_type == TL_PREC_JAC_BLOCK
        wrapteablockinit(x_min, x_max, y_min, y_max, halo_exchange_depth, cp, bfp, Kx, Ky, Di, rx, ry)
    else if preconditioner_type == TL_PREC_JAC_DIAG
        wrapteadiaginit(x_min, x_max, y_min, y_max, halo_exchange_depth, Mi, Kx, Ky, Di, rx, ry)
    end

    @threads for k in y_min:y_max
        for j in x_min:x_max
            w[j, k] = Di[j,k]*u[j, k] - ry*(Ky[j, k+1]*u[j, k+1] + Ky[j, k]*u[j, k-1]) - rx*(Kx[j+1, k]*u[j+1, k] + Kx[j, k]*u[j-1, k])

            # r[j, k] = u[j, k] # This is required to make a zero initial guess to match petsc errant behaviour
            # Only works one timestep is run
        end
    end
    r[:,:] = u - w
end

function tea_leaf_kernel_finalise!(x_min::Int, x_max::Int, y_min::Int, y_max::Int, halo_exchange_depth::Int, energy::Matrix{Float64}, density::Matrix{Float64}, u::Matrix{Float64})
    for param in [u, energy, density]
        @assert size(param) = (x_max+halo_exchange_depth - (x_min-halo_exchange_depth), y_max+halo_exchange_depth - (y_min-halo_exchange_depth))
    end

    # !$OMP PARALLEL
    # !$OMP DO
    #   DO k=y_min, y_max
    #     DO j=x_min, x_max
    #       energy(j,k) = u(j,k) / density(j,k)
    #     ENDDO
    #   ENDDO
    # !$OMP END DO
    # !$OMP END PARALLEL
    energy[:,:] = u ./ density
end

function tea_leaf_calc_residual_kernel!(x_min::Int, x_max::Int, y_min::Int, y_max::Int, halo_exchange_depth::Int, u::Matrix{Float64} , u0::Matrix{Float64}, r::Matrix{Float64}, Kx::Matrix{Float64}, Ky::Matrix{Float64}, Di::Matrix{Float64}, rx::Float64, ry::Float64)
    for param in [Kx, u, r, Ky, u0, Di]
        @assert size(param) = (x_max+halo_exchange_depth-(x_min-halo_exchange_depth),y_max+halo_exchange_depth-(y_min-halo_exchange_depth))
    end

    @threads for k in y_min:y_max
        for j in x_min:x_max
            smvp = Di(j,k)*u(j, k) - ry*(Ky(j, k+1)*u(j, k+1) + Ky(j, k)*u(j, k-1)) - rx*(Kx(j+1, k)*u(j+1, k) + Kx(j, k)*u(j-1, k))
            r(j, k) = u0(j, k) - smvp
        end
    end
end

# NOTE: you *must* use the result of this - it won't mutate the norm kernel
function tea_leaf_calc_2norm_kernel!(x_min::Int, x_max::Int, y_min::Int, y_max::Int, halo_exchange_depth::Int, arr::Matrix{Float64}, _::Float64)::Float64
    @assert size(arr) == (x_max+halo_exchange_depth-(x_min-halo_exchange_depth),y_max+halo_exchange_depth-(y_min-halo_exchange_depth))

    # !$OMP PARALLEL
    # !$OMP DO REDUCTION(+:norm)
    #     DO k=y_min,y_max
    #         DO j=x_min,x_max
    #             norm = norm + arr(j, k)*arr(j, k)
    #         ENDDO
    #     ENDDO
    # !$OMP END DO
    # !$OMP END PARALLEL
    return sum(arr .^ 2)
end

function tea_diag_init!(x_min::Int, x_max::Int, y_min::Int, y_max::Int, halo_exchange_depth::Int, Mi::Matrix{Float64}, Kx::Matrix{Float64}, Ky::Matrix{Float64}, Di::Matrix{Float64}, rx::Float64, ry::Float64)
    for param in [Kx, Ky, Di, Mi]
        @assert size(param) == (x_max+halo_exchange_depth-(x_min-halo_exchange_depth),y_max+halo_exchange_depth-(y_min-halo_exchange_depth))
    end

    omega=1.0
    @threads for k in y_min-halo_exchange_depth+1:y_max+halo_exchange_depth-1
        for j in x_min-halo_exchange_depth+1:x_max+halo_exchange_depth-1
            if Di[j, k] != 0.0
                Mi[j, k] = omega/Di[j, k]
            else
                Mi[j, k] = 0.0_8
            end
        end
    end
end

function tea_diag_solve!(x_min::Int, x_max::Int, y_min::Int, y_max::Int, halo_exchange_depth::Int, depth::Int, r::Matrix{Float64}, z::Matrix{Float64}, Mi::Matrix{Float64})
  for param in [r, z, Mi]
      @assert size(param) == (x_max+halo_exchange_depth-(x_min-halo_exchange_depth),y_max+halo_exchange_depth-(y_min-halo_exchange_depth))
  end

  # !$OMP DO
  #     DO k=y_min-depth,y_max+depth
  #       DO j=x_min-depth,x_max+depth
  #         z(j, k) = Mi(j, k)*r(j, k)
  #       ENDDO
  #     ENDDO
  # !$OMP END DO
  xr = x_min-depth:x_max+depth
  yr = y_min-depth:y_max+depth
  z[xr, yr] = Mi[xr, yr] .* r[xr, yr]
end

function tea_block_init!(x_min::Int, x_max::Int, y_min::Int, y_max::Int, halo_exchange_depth::Int, cp::Matrix{Float64}, bfp::Matrix{Float64}, Kx::Matrix{Float64}, Ky::Matrix{Float64}, Di::Matrix{Float64}, rx::Float64, ry::Float64)
    for param in [Kx, Ky, Di]
        @assert size(param) == (x_max+halo_exchange_depth-(x_min-halo_exchange_depth),y_max+halo_exchange_depth-(y_min-halo_exchange_depth))
    end

    for param in [cp, bfp]
        @assert size(param) == (x_max - x_min,y_max - y_min)
    end

    @threads for ko in y_min:jac_block_size:y_max
        bottom = ko
        top = min(ko + jac_block_size - 1, y_max)

        for j in x_min:x_max
            k = bottom
            cp[j, k] = (-Ky[j, k+1] * ry) / Di[j, k]

            for k in bottom+1:top
                bfp[j, k] = 1.0/(Di[j,k] - (-Ky[j, k]*ry)*cp[j, k-1])
                cp[j, k] = (-Ky[j, k+1]*ry)*bfp[j, k]
            end
        end
    end
end

function tea_block_solve!(x_min::Int, x_max::Int, y_min::Int, y_max::Int, halo_exchange_depth::Int, r::Matrix{Float64}, z::Matrix{Float64}, cp::Matrix{Float64}, bfp::Matrix{Float64}, Kx::Matrix{Float64}, Ky::Matrix{Float64}, Di::Matrix{Float64}, rx::Float64, ry::Float64)
    for param in  [Kx, Ky, Di, r, z]
        @assert size(param) == (x_max+halo_exchange_depth-(x_min-halo_exchange_depth),y_max+halo_exchange_depth-(y_min-halo_exchange_depth))
    end

    for param in [cp, bfp]
        @assert size(param) == (x_max - x_min, y_max - y_min)
    end

    dp_l= zeros(jac_block_size-1)
    z_l = zeros(jac_block_size-1)

    k_extra = y_max - (y_max % kstep)

    @thread ko in y_min:kstep:k_extra
    upper_k = ko+kstep - jac_block_size

    for ki in ko:jac_block_size:upper_k
        bottom = ki
        top = ki+jac_block_size - 1

        for j in x_min:x_max
            k = bottom
            dp_l[k-bottom] = r[j, k]/Di[j, k]

            for k in bottom+1:top
                dp_l[k-bottom] = (r[j, k] - (-Ky[j, k]*ry)*dp_l[k-bottom-1])*bfp[j, k]
            end

            k=top
            z_l[k-bottom] = dp_l[k-bottom]

            for k in top-1:-1:bottom
                z_l[k-bottom] = dp_l[k-bottom] - cp[j, k]*z_l[k-bottom+1]
            end

            for k in bottom:top
                z[j, k] = z_l[k-bottom]
            end
        end
    end

    @threads for ki in k_extra+1:jac_block_size:y_max
        bottom = min(ki, y_max)
        top = min(ki+jac_block_size-1, y_max)

        for j in x_min:x_max
            k = bottom
            dp_l[k-bottom] = r[j, k]/Di[j, k]

            for k in bottom+1:top
                dp_l[k-bottom] = (r[j, k] - (-Ky[j, k]*ry)*dp_l[k-bottom-1])*bfp[j, k]
            end

            k=top
            z_l[k-bottom] = dp_l[k-bottom]

            for k in top-1:-1:bottom
                z_l[k-bottom] = dp_l[k-bottom] - cp[j, k]*z_l[k-bottom+1]
            end

            for k in bottom:top
                z[j, k] = z_l[k-bottom]
            end
        end
    end
end

end
