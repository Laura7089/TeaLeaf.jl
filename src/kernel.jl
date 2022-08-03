function set_chunk_data!(settings::Settings, chunk::Chunk)
    x_min = settings.grid_x_min + settings.dx * chunk.left
    y_min = settings.grid_y_min + settings.dy * chunk.bottom

    xᵢ = 2:chunk.x+1
    @. chunk.vertex_x[xᵢ] = x_min + settings.dx * (xᵢ - settings.halo_depth)
    yᵢ = 2:chunk.y+1
    @. chunk.vertex_y[yᵢ] = y_min + settings.dy * (yᵢ - settings.halo_depth)

    xᵢ = 2:chunk.x
    @. chunk.cell_x[xᵢ] = 0.5 * (vertex_x[xᵢ] + vertex_x[3:chunk.x+1])
    yᵢ = 2:chunk.y
    @. chunk.cell_y[yᵢ] = 0.5 * (vertex_y[yᵢ] + vertex_y[3:chunk.y+1])

    A = 2:chunk.x*chunk.y
    volume[A] .= settings.dx * settings.dy
    x_area[A] .= settings.dy
    y_area[A] .= settings.dx
end

function set_chunk_state!(
    chunk::Chunk, # TODO: mutable?
    num_states::Int,
    states::Vector{State},
)
    # Set the initial state
    init = 2:chunk.x*chunk.y
    chunk.energy0[init] .= states[1].energy
    chunk.density[init] .= states[1].density

    # Apply all of the states in turn
    for ss = 1:num_states
        for jj = 1:chunk.y
            for kk = 1:chunk.x
                apply_state = false

                if states[ss].geometry == RECTANGULAR
                    apply_state =
                        chunk.vertex_x[kk+1] >= states[ss].x_min &&
                        chunk.vertex_x[kk] < states[ss].x_max &&
                        chunk.vertex_y[jj+1] >= states[ss].y_min &&
                        chunk.vertex_y[jj] < states[ss].y_max
                elseif states[ss].geometry == CIRCULAR
                    radius = sqrt(
                        (chunk.cell_x[kk] - states[ss].x_min) *
                        (chunk.cell_x[kk] - states[ss].x_min) +
                        (chunk.cell_y[jj] - states[ss].y_min) *
                        (chunk.cell_y[jj] - states[ss].y_min),
                    )

                    apply_state = radius <= states[ss].radius
                elseif states[ss].geometry == POINT
                    apply_state =
                        chunk.vertex_x[kk] == states[ss].x_min &&
                        chunk.vertex_y[jj] == states[ss].y_min
                end

                # Check if state applies at this vertex, and apply
                if apply_state
                    index = kk + jj * chunk.x
                    chunk.energy0[index] = states[ss].energy
                    chunk.density[index] = states[ss].density
                end
            end
        end
    end

    # Set an initial state for u
    index = @. (1:chunk.x-1) + (3:chunk.y-1) * chunk.x
    @. chunk.u[index] = chunk.energy0[index] * chunk.density[index]
end

# The kernel for updating halos locally
function local_halos!(chunk::Chunk, settings::Settings)
    for (index, buffer) in [
        (FIELD_DENSITY, density),
        (FIELD_P, p),
        (FIELD_ENERGY0, energy0),
        (FIELD_ENERGY1, energy),
        (FIELD_U, u),
        (FIELD_SD, sd),
    ]
        if settings.fields_to_exchange[index]
            update_face!(chunk, settings.halo_depth, buffer) # Done
        end
    end

end

# Updates faces in turn.
function update_face!(chunk::Chunk, halo_depth::Int, buffer::Vector{Float64})
    for (face, updatekernel) in [
        (CHUNK_LEFT, update_left!),
        (CHUNK_RIGHT, update_right!),
        (CHUNK_TOP, update_top!),
        (CHUNK_BOTTOM, update_bottom!),
    ]
        if chunk.chunk_neighbours[face] == EXTERNAL_FACE
            updatekernel(chunk, halo_depth, buffer)
        end
    end

end

# Update left halo.
function update_left!(chunk::Chunk, halo_depth::Int, buffer::Vector{Float64})
    base = (halo_depth+1:chunk.y-halo_depth) .* chunk.x
    kk = 2:chunk.depth
    @. buffer[base+(halo_depth-kk-1)] = buffer[base+(halo_depth+kk)]
end

# Update right halo.
function update_right!(chunk::Chunk, halo_depth::Int, buffer::Vector{Float64})
    base = (halo_depth+1:chunk.y-halo_depth) .* chunk.x
    kk = 2:chunk.depth
    @. buffer[base+(chunk.x-halo_depth+kk)] = buffer[base+(chunk.x-halo_depth-1-kk)]
end

# Update top halo.
function update_top!(chunk::Chunk, halo_depth::Int, buffer::Vector{Float64})
    jj = 2:chunk.depth
    base = halo_depth+1:chunk.x-halo_depth
    @. buffer[base+(chunk.y-halo_depth+jj)*chunk.x] =
        buffer[base+(chunk.y-halo_depth-1-jj)*chunk.x]
end

# Updates bottom halo.
function update_bottom!(chunk::Chunk, halo_depth::Int, buffer::Vector{Float64})
    jj = 2:chunk.depth
    base = halo_depth+1:chunk.x-halo_depth
    @. buffer[base+(halo_depth-jj-1)*chunk.x] = buffer[base+(halo_depth+jj)*chunk.x]
end

# Either packs or unpacks data from/to buffers.
function pack_or_unpack(
    chunk::Chunk,
    depth::Int,
    halo_depth::Int,
    face::Int,
    pack::Bool,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    using Match

#! format: off

     kernel = @match face begin
         CHUNK_LEFT => pack ? pack_left : unpack_left
         CHUNK_RIGHT => pack ? pack_right : unpack_right
         CHUNK_TOP => pack ? pack_top : unpack_top
         CHUNK_BOTTOM => pack ? pack_bottom : unpack_bottom
         - => throw("Incorrect face provided: $(face).")
     end

#! format: on
    kernel(x, y, depth, halo_depth, field, buffer)
end

# Packs left data into buffer.
function pack_left(
    x::Int,
    y::Int,
    depth::Int,
    halo_depth::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    for jj = halo_depth+1:y-halo_depth
        for kk = halo_depth+1:halo_depth+depth
            bufIndex = (kk - halo_depth) + (jj - halo_depth) * depth
            buffer[bufIndex] = field[jj*x+kk]
        end
    end
end

# Packs right data into buffer.
function pack_right(
    x::Int,
    y::Int,
    depth::Int,
    halo_depth::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    for jj = halo_depth+1:y-halo_depth
        for kk = x-halo_depth-depth+1:x-halo_depth
            bufIndex = (kk - (x - halo_depth - depth)) + (jj - halo_depth) * depth
            buffer[bufIndex] = field[jj*x+kk]
        end
    end
end

# Packs top data into buffer.
function pack_top(
    x::Int,
    y::Int,
    depth::Int,
    halo_depth::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    x_inner = x - 2 * halo_depth

    for jj = y-halo_depth-depth+1:y-halo_depth
        for kk = halo_depth+1:x-halo_depth
            bufIndex = (kk - halo_depth) + (jj - (y - halo_depth - depth)) * x_inner
            buffer[bufIndex] = field[jj*x+kk]
        end
    end
end

# Packs bottom data into buffer.
function pack_bottom(
    x::Int,
    y::Int,
    depth::Int,
    halo_depth::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    x_inner = x - 2 * halo_depth

    for jj = halo_depth+1:halo_depth+depth
        for kk = halo_depth+1:x-halo_depth
            bufIndex = (kk - halo_depth) + (jj - halo_depth) * x_inner
            buffer[bufIndex] = field[jj*x+kk]
        end
    end
end

# Unpacks left data from buffer.
function unpack_left(
    x::Int,
    y::Int,
    depth::Int,
    halo_depth::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    for jj = halo_depth+1:y-halo_depth
        for kk = halo_depth-depth+1:halo_depth
            bufIndex = (kk - (halo_depth - depth)) + (jj - halo_depth) * depth
            field[jj*x+kk] = buffer[bufIndex]
        end
    end
end

# Unpacks right data from buffer.
function unpack_right(
    x::Int,
    y::Int,
    depth::Int,
    halo_depth::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    for jj = halo_depth+1:y-halo_depth
        for kk = x-halo_depth+1:x-halo_depth+depth
            bufIndex = (kk - (x - halo_depth)) + (jj - halo_depth) * depth
            field[jj*x+kk] = buffer[bufIndex]
        end
    end
end

# Unpacks top data from buffer.
function unpack_top(
    x::Int,
    y::Int,
    depth::Int,
    halo_depth::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    x_inner = x - 2 * halo_depth

    for jj = y-halo_depth+1:y-halo_depth+depth
        for kk = halo_depth+1:x-halo_depth
            bufIndex = (kk - halo_depth) + (jj - (y - halo_depth)) * x_inner
            field[jj*x+kk] = buffer[bufIndex]
        end
    end
end

# Unpacks bottom data from buffer.
function unpack_bottom(
    x::Int,
    y::Int,
    depth::Int,
    halo_depth::Int,
    field::Vector{Float64},
    buffer::Vector{Float64},
)
    x_inner = x - 2 * halo_depth

    for jj = halo_depth-depth+1:halo_depth
        for kk = halo_depth+1:x-halo_depth
            bufIndex = (kk - halo_depth) + (jj - (halo_depth - depth)) * x_inner
            field[jj*x+kk] = buffer[bufIndex]
        end
    end
end

# Store original energy state
function store_energy(chunk::Chunk)
    for ii = 2:chunk.x*chunk.y
        chunk.energy[ii] = chunk.energy0[ii]
    end
end

# The field summary kernel
function field_summary(
    chunk::Chunk,
    halo_depth::Int,
    vol::Vector{Float64},
    mass::Vector{Float64},
    ie::Vector{Float64},
    temp::Vector{Float64},
)
    for jj = halo_depth+1:chunk.y-halo_depth
        for kk = halo_depth+1:chunk.x-halo_depth
            index = kk + jj * chunk.x
            cellVol = chunk.volume[index]
            cellMass = cellVol * chunk.density[index]
            vol += cellVol
            mass += cellMass
            ie += cellMass * energy0[index]
            temp += cellMass * u[index]
        end
    end

    return (vol, ie, temp, mass)
end

# Initialises the CG solver
function cg_init(
    chunk::Chunk,
    halo_depth::Int,
    coefficient::Int,
    rx::Float64,
    ry::Float64,
    rro::Float64,
)
    if coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY
        throw("Coefficient $(coefficient) is not valid.")
    end

    for jj = 2:chunk.y
        for kk = 2:chunk.x
            index = kk + jj * chunk.x
            chunk.p[index] = 0.0
            chunk.r[index] = 0.0
            chunk.u[index] = chunk.energy[index] * chunk.density[index]
        end
    end

    for jj = 3:chunk.y-1
        for kk = 3:chunk.x-1
            index = kk + jj * chunk.x
            chunk.w[index] =
                (coefficient == CONDUCTIVITY) ? chunk.density[index] :
                1.0 / chunk.density[index]
        end
    end

    for jj = halo_depth+1:chunk.y-1
        for kk = halo_depth+1:chunk.x-1
            index = kk + jj * chunk.x
            chunk.kx[index] =
                rx * (chunk.w[index-1] + chunk.w[index]) /
                (2.0 * chunk.w[index-1] * chunk.w[index])
            chunk.ky[index] =
                ry * (chunk.w[index-x] + chunk.w[index]) /
                (2.0 * chunk.w[index-x] * chunk.w[index])
        end
    end

    rro_temp = 0.0

    for jj = halo_depth+1:chunk.y-halo_depth
        for kk = halo_depth+1:chunk.x-halo_depth
            index = kk + jj * chunk.x
            smvp = SMVP(chunk.u)
            chunk.w[index] = smvp
            chunk.r[index] = chunk.u[index] - chunk.w[index]
            chunk.p[index] = chunk.r[index]
            rro_temp += chunk.r[index] * chunk.p[index]
        end
    end

    # Sum locally
    return rro + rro_temp
end

# Calculates w
function cg_calc_w(chunk::Chunk, halo_depth::Int, pw::Float64)
    pw_temp = 0.0

    for jj = halo_depth+1:chunk.y-halo_depth
        for kk = halo_depth+1:chunk.x-halo_depth
            index = kk + jj * chunk.x
            smvp = SMVP(chunk.p)
            chunk.w[index] = smvp
            pw_temp += chunk.w[index] * chunk.p[index]
        end
    end

    return pw + pw_temp
end

# Calculates u and r
function cg_calc_ur(chunk::Chunk, halo_depth::Int, alpha::Float64, rrn::Float64)
    rrn_temp = 0.0

    for jj = halo_depth+1:chunk.y-halo_depth
        for kk = halo_depth+1:chunk.x-halo_depth
            index = kk + jj * chunk.x

            chunk.u[index] += alpha * chunk.p[index]
            chunk.r[index] -= alpha * chunk.w[index]
            rrn_temp += chunk.r[index] * chunk.r[index]
        end
    end

    return rrn + rrn_temp
end

# Calculates p
function cg_calc_p(chunk::Chunk, halo_depth::Int, beta::Float64)
    for jj = halo_depth+1:chunk.y-halo_depth
        for kk = halo_depth+1:chunk.x-halo_depth
            index = kk + jj * chunk.x

            chunk.p[index] = beta * chunk.p[index] + chunk.r[index]
        end
    end
end

# Calculates the new value for u.
function cheby_calc_u(
    x::Int,
    y::Int,
    halo_depth::Int,
    u::Vector{Float64},
    p::Vector{Float64},
)
    for jj = halo_depth+1:y-halo_depth
        for kk = halo_depth+1:x-halo_depth
            index = kk + jj * x
            u[index] += p[index]
        end
    end
end

# Initialises the Chebyshev solver
function cheby_init(chunk::Chunk, halo_depth::Int)
    for jj = halo_depth+1:chunk.y-halo_depth
        for kk = halo_depth+1:chunk.x-halo_depth
            index = kk + jj * chunk.x
            smvp = SMVP(chunk.u)
            chunk.w[index] = smvp
            chunk.r[index] = chunk.u0[index] - chunk.w[index]
            chunk.p[index] = chunk.r[index] / chunk.theta
        end
    end

    cheby_calc_u(chunk.x, chunk.y, halo_depth, chunk.u, chunk.p) # Done
end

# The main chebyshev iteration
function cheby_iterate(chunk::Chunk, alpha::Float64, beta::Float64)
    for jj = halo_depth+1:chunk.y-halo_depth
        for kk = halo_depth+1:chunk.x-halo_depth
            index = kk + jj * chunk.x
            smvp = SMVP(chunk.u)
            chunk.w[index] = smvp
            chunk.r[index] = chunk.u0[index] - chunk.w[index]
            chunk.p[index] = alpha * chunk.p[index] + beta * chunk.r[index]
        end
    end

    cheby_calc_u(chunk.x, chunk.y, halo_depth, chunk.u, chunk.p) # Done
end

# Initialises the Jacobi solver
function jacobi_init(
    chunk::Chunk,
    halo_depth::Int,
    coefficient::Int,
    rx::Float64,
    ry::Float64,
)
    if coefficient < CONDUCTIVITY && coefficient < RECIP_CONDUCTIVITY
        throw("Coefficient $(coefficient) is not valid.")
    end

    index = @. (3:chunk.x-1) + (3:chunk.y-1) * chunk.x
    temp = chunk.energy[index] .* chunk.density[index]
    chunk.u0[index] .= temp
    chunk.u[index] .= temp

    for jj = halo_depth+1:chunk.y-1
        for kk = halo_depth+1:chunk.x-1
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
end

# The main Jacobi solve step
function jacobi_iterate(chunk::Chunk, halo_depth::Int)
    index = @. (2:chunk.x) + (2:chunk.y) * chunk.x
    chunk.r[index] .= chunk.u[index]

    err = 0.0
    for jj = halo_depth+1:chunk.y-halo_depth
        for kk = halo_depth+1:chunk.x-halo_depth
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
    end

    return err
end

# Initialises the PPCG solver
function ppcg_init(chunk::Chunk, halo_depth::Int)
    index = (halo_depth+1:chunk.x-halo_depth) + (halo_depth+1:chunk.y-halo_depth) * chunk.x
    @. chunk.sd[index] = chunk.r[index] / chunk.theta
end

# The PPCG inner iteration
function ppcg_inner_iteration(chunk::Chunk, halo_depth::Int, alpha::Float64, beta::Float64)
    index = (halo_depth+1:chunk.x-halo_depth) + (halo_depth+1:chunk.y-halo_depth) * chunk.x

    smvp = SMVP(chunk.sd)
    chunk.r[index] .-= smvp
    chunk.u[index] .+= chunk.sd[index]

    @. chunk.sd[index] = alpha * chunk.sd[index] + beta * chunk.r[index]
end

# Copies the current u into u0
function copy_u(chunk::Chunk, halo_depth::Int)
    index = (halo_depth+1:chunk.x-halo_depth) + (halo_depth+1:chunk.y-halo_depth) * chunk.x
    chunk.u0[index] .= chunk.u[index]
end

# Calculates the current value of r
function calculate_residual(chunk::Chunk, halo_depth::Int)
    for jj = halo_depth+1:chunk.y-halo_depth
        for kk = halo_depth+1:chunk.x-halo_depth
            index = kk + jj * chunk.x
            smvp = SMVP(chunk.u)
            chunk.r[index] = chunk.u0[index] - smvp
        end
    end
end

# Calculates the 2 norm of a given buffer
function calculate_2norm(
    chunk::Chunk,
    halo_depth::Int,
    buffer::Vector{Float64},
    norm::Float64,
)
    index = (halo_depth+1:chunk.x-halo_depth) + (halo_depth+1:chunk.y-halo_depth) * chunk.x
    return sum(buffer[index] .^ 2) + norm
end

# Finalises the solution
function finalise(chunk::Chunk, halo_depth::Int)
    index = (halo_depth+1:chunk.x-halo_depth) + (halo_depth+1:chunk.y-halo_depth) * chunk.x
    @. chunk.energy[index] = chunk.u[index] / chunk.density[index]
end
