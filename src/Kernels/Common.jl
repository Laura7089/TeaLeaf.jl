module Common

export tea_leaf_common_init_kernel

CHUNK_LEFT::Int   =1
CHUNK_RIGHT::Int  =2
CHUNK_BOTTOM::Int =3
CHUNK_TOP::Int    =4
EXTERNAL_FACE::Int=-1

TL_PREC_NONE::Int       = 1
TL_PREC_JAC_DIAG::Int   = 2
TL_PREC_JAC_BLOCK::Int  = 3

CONDUCTIVITY::Int        = 1
RECIP_CONDUCTIVITY::Int  = 2

jac_block_size::Int = 4
block_size::Int =1
kstep::Int = block_size*jac_block_size

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
        # Threads.@threads for k in (y_min-halo_exchange_depth):(y_max+halo_exchange_depth)
        #     for j in x_min-halo_exchange_depth:(x_max+halo_exchange_depth)
        #         w[j,k]=1.0/density[j  ,k  ]
        #     end
        # end
        w[:,:] = 1.0 ./ density
    else if coef == CONDUCTIVITY
        # Threads.@threads for k in (y_min-halo_exchange_depth):(y_max+halo_exchange_depth)
        #     for j in (x_min-halo_exchange_depth):(x_max+halo_exchange_depth)
        #         w[j  ,k  ]=density[j  ,k  ]
        #     end
        # end
        w[:,:] = density
    end

    Threads.@threads for k in (y_min-halo_exchange_depth + 1):(y_max+halo_exchange_depth)
        for j in (x_min-halo_exchange_depth + 1):(x_max+halo_exchange_depth)
            Kx[j,k]=(w[j-1,k  ] + w[j,k])/(2.0*w[j-1,k  ]*w[j,k])
            Ky[j,k]=(w[j  ,k-1] + w[j,k])/(2.0*w[j  ,k-1]*w[j,k])
        end
    end

    # Whether to apply reflective boundary conditions to all external faces
    if !reflective_boundary
        if zero_boundary[CHUNK_LEFT]
            # Threads.@threads for k in (y_min-halo_exchange_depth):(y_max+halo_exchange_depth)
            #     for j in (x_min-halo_exchange_depth):x_min
            #         Kx[j,k]=0.0
            #     end
            # end
            Kx[(x_min - halo_exchange_depth):x_min,:] = zeros()
        end

        if zero_boundary[CHUNK_RIGHT]
            # Threads.@threads for  k in (y_min-halo_exchange_depth):(y_max+halo_exchange_depth)
            #     for j in (x_max + 1):(x_max+halo_exchange_depth)
            #         Kx[j,k]=0.0
            #     end
            # end
            Kx[(x_max + 1):(x_max + halo_exchange_depth),:] = zeros()
        en

        if zero_boundary[CHUNK_BUTTOM]
            # Threads.@threads for k in (y_min-halo_exchange_depth):y_min
            #     for j in (x_min-halo_exchange_depth):(x_max+halo_exchange_depth)
            #         Ky[j,k]=0.0
            #     end
            # end
            Kx[:,(y_min-halo_exchange_depth):y_min] = zeros()
        end

        if zero_boundary[CHUNK_TOP]
            # Threads.@threads for k in (y_max + 1):(y_max+halo_exchange_depth)
            #     for j in (x_min-halo_exchange_depth):(x_max+halo_exchange_depth)
            #         Ky[j,k]=0.0
            #     end
            # end
            Kx[(x_min-halo_exchange_depth):(x_max+halo_exchange_depth),(y_max + 1):(y_max+halo_exchange_depth)] = zeros()
        end
    end

    # Setup storage for the diagonal entries
    Threads.@threads for k in (y_min-halo_exchange_depth+1):(y_max+halo_exchange_depth-1)
        for j in (x_min-halo_exchange_depth+1):(x_max+halo_exchange_depth-1)
            Di[j,k]=(1.0+ ry*(Ky[j, k+1] + Ky[j, k])+ rx*(Kx[j+1, k] + Kx[j, k]))
        end
    end

    if preconditioner_type == TL_PREC_JAC_BLOCK
        wrapteablockinit(x_min, x_max, y_min, y_max, halo_exchange_depth, cp, bfp, Kx, Ky, Di, rx, ry)
    else if preconditioner_type == TL_PREC_JAC_DIAG
        wrapteadiaginit(x_min, x_max, y_min, y_max, halo_exchange_depth, Mi, Kx, Ky, Di, rx, ry)
    end

    Threads.@threads for k in y_min:y_max
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

end
