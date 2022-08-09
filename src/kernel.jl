# Updates faces in turn.
function update_face!(chunk::Chunk, hd::Int, depth::Int, buffer::Vector{Float64})
    for (face, updatekernel) in [
        (CHUNK_LEFT, update_left!),
        (CHUNK_RIGHT, update_right!),
        (CHUNK_TOP, update_top!),
        (CHUNK_BOTTOM, update_bottom!),
    ]
        if chunk.neighbours[face] == EXTERNAL_FACE
            updatekernel(chunk, hd, depth, buffer)
        end
    end
end

# Update left halo.
function update_left!(chunk::Chunk, hd::Int, depth::Int, buffer::Vector{Float64})
    for jj = hd+1:chunk.y-hd, kk = 1:depth
        base = jj * chunk.x
        buffer[base+hd-kk-1] = buffer[base+hd+kk]
    end
end

# Update right halo.
function update_right!(chunk::Chunk, hd::Int, depth::Int, buffer::Vector{Float64})
    for jj = hd:chunk.y-hd-1, kk = 1:depth
        base = jj * chunk.x
        buffer[base+(chunk.x-hd+kk)] = buffer[base+(chunk.x-hd-1-kk)]
    end
end

# Update top halo.
function update_top!(chunk::Chunk, hd::Int, depth::Int, buffer::Vector{Float64})
    for jj = 1:depth, kk = hd+1:chunk.x-hd
        buffer[kk+(chunk.y-hd+jj)*chunk.x] = buffer[kk+(chunk.y-hd-1-jj)*chunk.x]
    end
end

# Updates bottom halo.
function update_bottom!(chunk::Chunk, hd::Int, depth::Int, buffer::Vector{Float64})
    for jj = 1:depth, kk = hd+1:chunk.x-hd
        @debug "" kk+(hd-jj)*chunk.x kk+(hd+jj)*chunk.x chunk.x depth
        buffer[kk+(hd-jj)*chunk.x] = buffer[kk+(hd+jj)*chunk.x]
    end
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
    for kk in (hd:chunk.y-hd-1), jj in (hd+1:chunk.x-hd)
        index = jj + kk * chunk.x
        smvp = SMVP(chunk, chunk.u, index)
        chunk.r[index] = chunk.u0[index] - smvp
    end
end

# Calculates the 2 norm of a given buffer
function calculate_2norm(chunk::Chunk, hd::Int, buffer::Vector{Float64})
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    return sum(buffer[index] .^ 2)
end

# Finalises the solution
function finalise!(chunk::Chunk, hd::Int)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    @. chunk.energy[index] = chunk.u[index] / chunk.density[index]
end
