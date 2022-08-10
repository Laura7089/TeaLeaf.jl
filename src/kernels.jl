module Kernels

function field_summary(
    chunk::C,
    settings::Settings,
    is_solve_finished::Bool,
) where {C<:Chunk}
    vol = 0.0
    ie = 0.0
    temp = 0.0
    mass = 0.0

    for jj = settings.halo_depth+1:chunk.y-settings.halo_depth,
        kk = settings.halo_depth+1:chunk.x-settings.halo_depth

        cellVol = chunk.volume[kk, jj]
        cellMass = cellVol * chunk.density[kk, jj]
        vol += cellVol
        mass += cellMass
        ie += cellMass * chunk.energy0[kk, jj]
        temp += cellMass * chunk.u[kk, jj]
    end

    if settings.check_result && is_solve_finished
        @info "Checking results..."
        checking_value = get_checking_value(settings) # Done
        @info "Expected and actual:" checking_value temp

        qa_diff = abs(100.0 * (temp / checking_value) - 100.0)
        if qa_diff < 0.001
            @info "This run PASSED" qa_diff
        else
            @warn "This run FAILED" qa_diff
        end
    end
end

# Invoke the halo update kernels
function halo_update!(chunk::C, settings::Settings, depth::Int) where {C<:Chunk}
    # Check that we actually have exchanges to perform
    if !any(settings.fields_to_exchange)
        return
    end

    for (index, buffer) in [
        (FIELD_DENSITY, :density),
        (FIELD_P, :p),
        (FIELD_ENERGY0, :energy0),
        (FIELD_ENERGY1, :energy),
        (FIELD_U, :u),
        (FIELD_SD, :sd),
    ]

        if settings.fields_to_exchange[index]
            update_face!(chunk, settings.halo_depth, depth, getfield(chunk, buffer))
        end
    end
end

# Calls all kernels that wrap up a solve regardless of solver
function solve_finished!(chunk::C, settings::Settings) where {C<:Chunk}
    exact_error = 0.0

    if settings.check_result
        calculate_residual!(chunk, settings.halo_depth)

        exact_error += calculate_2norm(chunk, settings.halo_depth, chunk.r)
    end

    finalise!(chunk, settings.halo_depth)
    settings.fields_to_exchange[FIELD_ENERGY1] = true
    halo_update!(chunk, settings, 1)
end

# Sparse Matrix Vector Product
function smvp(chunk::C, a::Matrix{Float64}, x::Int, y::Int)::Float64 where {C<:Chunk}
    (1 + chunk.kx[x+1, y] + chunk.kx[x, y] + chunk.ky[x, y+1] + chunk.ky[x, y]) * a[x, y]
    -(chunk.kx[x+1, y] * a[x+1, y] + chunk.kx[x, y] * a[x-1, y])
    -(chunk.ky[x, y+1] * a[x, y+1] + chunk.ky[x, y] * a[x, y-1])
end

# Updates faces in turn.
function update_face!(
    chunk::C,
    hd::Int,
    depth::Int,
    buffer::B,
) where {C<:Chunk,B<:AbstractMatrix{Float64}}
    for (face, updatekernel) in [
        (CHUNK_LEFT, update_left!),
        (CHUNK_RIGHT, update_right!),
        (CHUNK_TOP, update_top!),
        (CHUNK_BOTTOM, update_bottom!),
    ]
        # TODO: is this necessary without MPI?
        updatekernel(chunk, hd, depth, buffer)
    end
end

# Update left halo.
function update_left!(
    chunk::C,
    hd::Int,
    depth::Int,
    buffer::B,
) where {C<:Chunk,B<:AbstractMatrix{Float64}}
    for jj = hd+1:chunk.y-hd, kk = 1:depth
        # TODO: is the +1 right?
        buffer[hd-kk+1, jj] = buffer[hd+kk, jj]
    end
end

# Update right halo.
function update_right!(
    chunk::C,
    hd::Int,
    depth::Int,
    buffer::B,
) where {C<:Chunk,B<:AbstractMatrix{Float64}}
    for jj = hd:chunk.y-hd-1, kk = 1:depth
        buffer[chunk.x-hd+kk, jj] = buffer[chunk.x-hd-1-kk, jj]
    end
end

# Update top halo.
function update_top!(
    chunk::C,
    hd::Int,
    depth::Int,
    buffer::B,
) where {C<:Chunk,B<:AbstractMatrix{Float64}}
    for jj = 1:depth, kk = hd+1:chunk.x-hd
        buffer[kk, chunk.y-hd+jj] = buffer[kk, chunk.y-hd-1-jj]
    end
end

# Updates bottom halo.
function update_bottom!(
    chunk::C,
    hd::Int,
    depth::Int,
    buffer::B,
) where {C<:Chunk,B<:AbstractMatrix{Float64}}
    for jj = 1:depth, kk = hd+1:chunk.x-hd
        # TODO: is the +1 right?
        buffer[kk, hd-jj+1] = buffer[kk, hd+jj]
    end
end

# Either packs or unpacks data from/to buffers.
function pack_or_unpack(
    chunk::C,
    depth::Int,
    hd::Int,
    face::Int,
    pack::Bool,
    field::Vector{Float64},
    buffer::Vector{Float64},
) where {C<:Chunk}
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
function store_energy!(chunk::Chunk)
    chunk.energy .= chunk.energy0
end

# Copies the current u into u0
function copy_u!(chunk::Chunk, hd::Int)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    chunk.u0[index] .= chunk.u[index]
end

# Calculates the current value of r
function calculate_residual!(chunk::Chunk, hd::Int)
    for kk in (hd+1:chunk.y-hd-1), jj in (hd+1:chunk.x-hd)
        p = smvp(chunk, chunk.u, kk, jj)
        chunk.r[kk, jj] = chunk.u0[kk, jj] - p
    end
end

# Calculates the 2 norm of a given buffer
function calculate_2norm(chunk::Chunk, hd::Int, buffer::AbstractMatrix{Float64})
    return sum(buffer[hd+1:chunk.x-hd, hd+1:chunk.y-hd] .^ 2)
end

# Finalises the solution
function finalise!(chunk::Chunk, hd::Int)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    @. chunk.energy[index] = chunk.u[index] / chunk.density[index]
end

end
