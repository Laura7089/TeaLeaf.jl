# Updates faces in turn.
function update_face!(chunk::Chunk, hd::Int, buffer::Vector{Float64})
    for (face, updatekernel) in [
        (CHUNK_LEFT, update_left!),
        (CHUNK_RIGHT, update_right!),
        (CHUNK_TOP, update_top!),
        (CHUNK_BOTTOM, update_bottom!),
    ]
        if chunk.chunk_neighbours[face] == EXTERNAL_FACE
            updatekernel(chunk, hd, buffer)
        end
    end
end

# Update left halo.
function update_left!(chunk::Chunk, hd::Int, buffer::Vector{Float64})
    base = (hd+1:chunk.y-hd) .* chunk.x
    kk = 2:chunk.depth
    @. buffer[base+(hd-kk-1)] = buffer[base+(hd+kk)]
end

# Update right halo.
function update_right!(chunk::Chunk, hd::Int, buffer::Vector{Float64})
    base = (hd+1:chunk.y-hd) .* chunk.x
    kk = 2:chunk.depth
    @. buffer[base+(chunk.x-hd+kk)] = buffer[base+(chunk.x-hd-1-kk)]
end

# Update top halo.
function update_top!(chunk::Chunk, hd::Int, buffer::Vector{Float64})
    jj = 2:chunk.depth
    base = hd+1:chunk.x-hd
    @. buffer[base+(chunk.y-hd+jj)*chunk.x] = buffer[base+(chunk.y-hd-1-jj)*chunk.x]
end

# Updates bottom halo.
function update_bottom!(chunk::Chunk, hd::Int, buffer::Vector{Float64})
    jj = 2:chunk.depth
    base = hd+1:chunk.x-hd
    @. buffer[base+(hd-jj-1)*chunk.x] = buffer[base+(hd+jj)*chunk.x]
end

# Either packs or unpacks data from/to buffers.
function pack_or_unpack(
    chunk::Chunk,
    depth::Int,
    hd::Int,
    face::Int,
    pack::Bool,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    kernel = @match (face, pack) begin
        (CHUNK_LEFT, true) => pack_left
        (CHUNK_LEFT, false) => unpack_left
        (CHUNK_RIGHT, true) => pack_right
        (CHUNK_RIGHT, false) => unpack_right
        (CHUNK_TOP, true) => pack_top
        (CHUNK_TOP, false) => unpack_top
        (CHUNK_BOTTOM, true) => pack_bottom
        (CHUNK_BOTTOM, false) => unpack_bottom
        _ => throw("Incorrect face provided: $(face).")
    end
    kernel(x, y, depth, hd, field, buffer)
end

# Packs left data into buffer.
function pack_left(
    x::Int,
    y::Int,
    depth::Int,
    hd::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    for jj = hd+1:y-hd, kk = hd+1:hd+depth
        bufIndex = (kk - hd) + (jj - hd) * depth
        buffer[bufIndex] = field[jj*x+kk]
    end
end

# Packs right data into buffer.
function pack_right(
    x::Int,
    y::Int,
    depth::Int,
    hd::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    for jj = hd+1:y-hd, kk = x-hd-depth+1:x-hd
        bufIndex = (kk - (x - hd - depth)) + (jj - hd) * depth
        buffer[bufIndex] = field[jj*x+kk]
    end
end

# Packs top data into buffer.
function pack_top(
    x::Int,
    y::Int,
    depth::Int,
    hd::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    x_inner = x - 2 * hd

    for jj = y-hd-depth+1:y-hd, kk = hd+1:x-hd
        bufIndex = (kk - hd) + (jj - (y - hd - depth)) * x_inner
        buffer[bufIndex] = field[jj*x+kk]
    end
end

# Packs bottom data into buffer.
function pack_bottom(
    x::Int,
    y::Int,
    depth::Int,
    hd::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    x_inner = x - 2 * hd

    for jj = hd+1:hd+depth, kk = hd+1:x-hd
        bufIndex = (kk - hd) + (jj - hd) * x_inner
        buffer[bufIndex] = field[jj*x+kk]
    end
end

# Unpacks left data from buffer.
function unpack_left(
    x::Int,
    y::Int,
    depth::Int,
    hd::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    for jj = hd+1:y-hd, kk = hd-depth+1:hd
        bufIndex = (kk - (hd - depth)) + (jj - hd) * depth
        field[jj*x+kk] = buffer[bufIndex]
    end
end

# Unpacks right data from buffer.
function unpack_right(
    x::Int,
    y::Int,
    depth::Int,
    hd::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    for jj = hd+1:y-hd, kk = x-hd+1:x-hd+depth
        bufIndex = (kk - (x - hd)) + (jj - hd) * depth
        field[jj*x+kk] = buffer[bufIndex]
    end
end

# Unpacks top data from buffer.
function unpack_top(
    x::Int,
    y::Int,
    depth::Int,
    hd::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    x_inner = x - 2 * hd

    for jj = y-hd+1:y-hd+depth, kk = hd+1:x-hd
        bufIndex = (kk - hd) + (jj - (y - hd)) * x_inner
        field[jj*x+kk] = buffer[bufIndex]
    end
end

# Unpacks bottom data from buffer.
function unpack_bottom(
    x::Int,
    y::Int,
    depth::Int,
    hd::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    x_inner = x - 2 * hd

    for jj = hd-depth+1:hd, kk = hd+1:x-hd
        bufIndex = (kk - hd) + (jj - (hd - depth)) * x_inner
        field[jj*x+kk] = buffer[bufIndex]
    end
end

# Store original energy state
function store_energy(chunk::Chunk)
    for ii = 1:chunk.x*chunk.y
        chunk.energy[ii] = chunk.energy0[ii]
    end
end

# The field summary kernel
function field_summary(
    chunk::Chunk,
    hd::Int,
    vol::Vector{Float64},
    mass::Vector{Float64},
    ie::Vector{Float64},
    temp::Vector{Float64},
)
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
        cellVol = chunk.volume[index]
        cellMass = cellVol * chunk.density[index]
        vol += cellVol
        mass += cellMass
        ie += cellMass * energy0[index]
        temp += cellMass * u[index]
    end

    return (vol, ie, temp, mass)
end

# Initialises the CG solver
function cg_init(chunk::Chunk, hd::Int, coefficient::Int, rx::Float64, ry::Float64)
    if coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY
        throw("Coefficient $(coefficient) is not valid.")
    end

    for jj = 2:chunk.y, kk = 2:chunk.x
        index = kk + jj * chunk.x
        chunk.p[index] = 0.0
        chunk.r[index] = 0.0
        chunk.u[index] = chunk.energy[index] * chunk.density[index]
    end

    for jj = 3:chunk.y-1, kk = 3:chunk.x-1
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
            ry * (chunk.w[index-x] + chunk.w[index]) /
            (2.0 * chunk.w[index-x] * chunk.w[index])
    end

    rro_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
        smvp = SMVP(chunk.u)
        chunk.w[index] = smvp
        chunk.r[index] = chunk.u[index] - chunk.w[index]
        chunk.p[index] = chunk.r[index]
        rro_temp += chunk.r[index] * chunk.p[index]
    end

    return rro_temp
end

# Calculates w
function cg_calc_w(chunk::Chunk, hd::Int)
    pw_temp = 0.0

    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
        smvp = SMVP(chunk.p)
        chunk.w[index] = smvp
        pw_temp += chunk.w[index] * chunk.p[index]
    end

    return pw_temp
end

# Calculates u and r
function cg_calc_ur(chunk::Chunk, hd::Int, alpha::Float64)
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
function cg_calc_p(chunk::Chunk, hd::Int, beta::Float64)
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x

        chunk.p[index] = beta * chunk.p[index] + chunk.r[index]
    end
end

# Calculates the new value for u.
function cheby_calc_u(x::Int, y::Int, hd::Int, u::Vector{Float64}, p::Vector{Float64})
    for jj = hd+1:y-hd, kk = hd+1:x-hd
        index = kk + jj * x
        u[index] += p[index]
    end
end

# Initialises the Chebyshev solver
function cheby_init(chunk::Chunk, hd::Int)
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
        smvp = SMVP(chunk.u)
        chunk.w[index] = smvp
        chunk.r[index] = chunk.u0[index] - chunk.w[index]
        chunk.p[index] = chunk.r[index] / chunk.theta
    end

    cheby_calc_u(chunk.x, chunk.y, hd, chunk.u, chunk.p) # Done
end

# The main chebyshev iteration
function cheby_iterate(chunk::Chunk, alpha::Float64, beta::Float64)
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
        smvp = SMVP(chunk.u)
        chunk.w[index] = smvp
        chunk.r[index] = chunk.u0[index] - chunk.w[index]
        chunk.p[index] = alpha * chunk.p[index] + beta * chunk.r[index]
    end

    cheby_calc_u(chunk.x, chunk.y, hd, chunk.u, chunk.p) # Done
end

# Initialises the Jacobi solver
function jacobi_init(chunk::Chunk, hd::Int, coefficient::Int, rx::Float64, ry::Float64)
    if coefficient < CONDUCTIVITY && coefficient < RECIP_CONDUCTIVITY
        throw("Coefficient $(coefficient) is not valid.")
    end

    index = @. (3:chunk.x-1) + (3:chunk.y-1) * chunk.x
    temp = chunk.energy[index] .* chunk.density[index]
    chunk.u0[index] .= temp
    chunk.u[index] .= temp

    for jj = hd+1:chunk.y-1, kk = hd+1:chunk.x-1
        index = kk + jj * chunk.x
        densityCentre =
            (coefficient == CONDUCTIVITY) ? chunk.density[index] :
            1.0 / chunk.density[index]
        densityLeft =
            (coefficient == CONDUCTIVITY) ? chunk.density[index-1] :
            1.0 / chunk.density[index-1]
        densityDown =
            (coefficient == CONDUCTIVITY) ? chunk.density[index-x] :
            1.0 / chunk.density[index-chunk.x]

        chunk.kx[index] =
            rx * (densityLeft + densityCentre) / (2.0 * densityLeft * densityCentre)
        chunk.ky[index] =
            ry * (densityDown + densityCentre) / (2.0 * densityDown * densityCentre)
    end
end

# The main Jacobi solve step
function jacobi_iterate(chunk::Chunk, hd::Int)
    index = @. (2:chunk.x) + (2:chunk.y) * chunk.x
    chunk.r[index] .= chunk.u[index]

    err = 0.0
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
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

# Initialises the PPCG solver
function ppcg_init(chunk::Chunk, hd::Int)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    @. chunk.sd[index] = chunk.r[index] / chunk.theta
end

# The PPCG inner iteration
function ppcg_inner_iteration(chunk::Chunk, hd::Int, alpha::Float64, beta::Float64)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x

    smvp = SMVP(chunk.sd)
    chunk.r[index] .-= smvp
    chunk.u[index] .+= chunk.sd[index]

    @. chunk.sd[index] = alpha * chunk.sd[index] + beta * chunk.r[index]
end

# Copies the current u into u0
function copy_u(chunk::Chunk, hd::Int)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    chunk.u0[index] .= chunk.u[index]
end

# Calculates the current value of r
function calculate_residual(chunk::Chunk, hd::Int)
    smvp = SMVP(chunk.u)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    @. chunk.r[index] = chunk.u0[index] - smvp
end

# Calculates the 2 norm of a given buffer
function calculate_2norm(chunk::Chunk, hd::Int, buffer::Vector{Float64}, norm::Float64)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    return sum(buffer[index] .^ 2) + norm
end

# Finalises the solution
function finalise(chunk::Chunk, hd::Int)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    @. chunk.energy[index] = chunk.u[index] / chunk.density[index]
end
