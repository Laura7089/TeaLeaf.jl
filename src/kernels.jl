module Kernels

using TeaLeaf
using Match
using ExportAll

const ERROR_START = 1e+10
const ERROR_SWITCH_MAX = 1.0

function fieldsummary(chunk::C, set::Settings, solvefin::Bool) where {C<:Chunk}
    x, y = size(chunk)
    kk = set.halodepth+1:x-set.halodepth
    jj = set.halodepth+1:y-set.halodepth
    temp = sum(chunk.volume[kk, jj] .* chunk.density[kk, jj] .* chunk.u[kk, jj])

    if set.checkresult && solvefin
        @info "Checking results..."
        cv = checkingvalue(set)
        @info "Expected and actual:" cv temp

        qa_diff = abs(100temp / cv - 100.0)
        if qa_diff < 0.001
            @info "This run PASSED" qa_diff
        else
            @warn "This run FAILED" qa_diff
        end
    end
end

# Invoke the halo update kernels
function haloupdate!(chunk::C, set::Settings, depth::Int) where {C<:Chunk}
    toexchange = filter(f -> set.toexchange[f], CHUNK_FIELDS)
    # Call `updateface` on all faces which are marked in `toexchange`
    updateface!.(chunk, set.halodepth, depth, getfield.(chunk, toexchange))
end

# Calls all kernels that wrap up a solve regardless of solver
function solvefinished!(chunk::C, set::Settings) where {C<:Chunk}
    # TODO this doesn't seem to do anything
    exact_error = 0.0

    if set.checkresult
        residual!(chunk, set.halodepth)

        exact_error += twonorm(chunk, set.halodepth, chunk.r)
    end

    finalise!(chunk, set.halodepth)
    set.toexchange[:energy1] = true
    haloupdate!(chunk, set, 1)
end

# Sparse Matrix Vector Product
# TODO: what the hell is this trying to do???
function smvp(
    chunk::C,
    a::AbstractMatrix{Float64},
    index::Tuple{Int,Int},
)::Float64 where {C<:Chunk}
    x, y = index
    # TODO: remove me
    for (i, arr) in enumerate([chunk.kx, a, chunk.ky])
        if any(isnan, arr)
            throw("NaN found: $((x, y)), $(i)")
        end
    end
    consum = sum((1, chunk.kx[x+1, y], chunk.kx[x, y], chunk.ky[x, y+1], chunk.ky[x, y]))
    return consum * a[x, y]
    -(chunk.kx[x+1, y] * a[x+1, y] + chunk.kx[x, y] * a[x-1, y])
    -(chunk.ky[x, y+1] * a[x, y+1] + chunk.ky[x, y] * a[x, y-1])
end

# Updates faces in turn.
function updateface!(
    chunk::C,
    hd::Int,
    depth::Int,
    buffer::B,
) where {C<:Chunk,B<:AbstractMatrix{Float64}}
    x, y = size(chunk)
    ys = hd+1:y-hd
    # Update left halo.
    for kk = 1:depth
        # TODO: is the +1 right?
        buffer[hd-kk+1, ys] .= buffer[hd+kk, ys]
    end
    # Update right halo.
    for kk = 1:depth
        buffer[x-hd+kk, ys] .= buffer[x-hd-1-kk, ys]
    end
    xs = hd+1:x-hd
    # Update top halo.
    for jj = 1:depth
        buffer[xs, y-hd+jj] .= buffer[xs, y-hd-1-jj]
    end
    # Updates bottom halo.
    for jj = 1:depth
        # TODO: is the +1 right?
        buffer[xs, hd-jj+1] .= buffer[xs, hd+jj]
    end
end

# Either packs or unpacks data from/to buffers.
function packunpack(
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
        f => throw("Incorrect face provided: $(f).")
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
        buffer[bufIndex] = field[kk, jj]
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
        buffer[bufIndex] = field[kk, jj]
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
    x_inner = x - 2hd

    for jj = y-hd-depth+1:y-hd, kk = hd+1:x-hd
        bufIndex = (kk - hd) + (jj - (y - hd - depth)) * x_inner
        buffer[bufIndex] = field[kk, jj]
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
    x_inner = x - 2hd

    for jj = hd+1:hd+depth, kk = hd+1:x-hd
        bufIndex = (kk - hd) + (jj - hd) * x_inner
        buffer[bufIndex] = field[kk, jj]
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
        field[kk, jj] = buffer[bufIndex]
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
        field[kk, jj] = buffer[bufIndex]
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
    x_inner = x - 2hd

    for jj = y-hd+1:y-hd+depth, kk = hd+1:x-hd
        bufIndex = (kk - hd) + (jj - (y - hd)) * x_inner
        field[kk, jj] = buffer[bufIndex]
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
    x_inner = x - 2hd

    for jj = hd-depth+1:hd, kk = hd+1:x-hd
        field[kk, jj] = buffer[kk-hd, jj-(hd-depth)]
    end
end

# Copies the current u into u0
function copyu!(chunk::Chunk, hd::Int)
    x, y = size(chunk)
    chunk.u0[hd+1:x-hd, hd+1:y-hd] .= chunk.u[hd+1:x-hd, hd+1:y-hd]
end

# Calculates the current value of r
function residual!(chunk::Chunk, hd::Int)
    x, y = size(chunk)
    for jj = hd+1:x-hd, kk = hd+1:y-hd
        p = smvp(chunk, chunk.u, (kk, jj))
        chunk.r[kk, jj] = chunk.u0[kk, jj] - p
    end
end

# Calculates the 2 norm of a given buffer
function twonorm(chunk::Chunk, hd::Int, buffer::AbstractMatrix{Float64})
    x, y = size(chunk)
    return sum(buffer[hd+1:x-hd, hd+1:y-hd] .^ 2)
end

# Finalises the solution
function finalise!(chunk::Chunk, hd::Int)
    x, y = size(chunk)
    @. chunk.energy[hd+1:x-hd, hd+1:y-hd] =
        chunk.u[hd+1:x-hd, hd+1:y-hd] / chunk.density[hd+1:x-hd, hd+1:y-hd]
end

@exportAll()

end
