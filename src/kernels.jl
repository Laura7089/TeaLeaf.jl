module Kernels

using TeaLeaf
using Match
using ExportAll
using LinearAlgebra: norm

const ERROR_START = 1e+10
const ERROR_SWITCH_MAX = 1.0

function fieldsummary(chunk::C, set::Settings, solvefin::Bool) where {C<:Chunk}
    H = halo(chunk, set.halodepth)
    temp = sum(chunk.volume[H] .* chunk.density[H] .* chunk.u[H])

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
        exact_error += norm(chunk.r[halo(chunk, set.halodepth)])
    end

    finalise!(chunk, set.halodepth)
    set.toexchange[:energy1] = true
    haloupdate!(chunk, set, 1)
end

# Sparse Matrix Vector Product
# TODO: what the hell is this trying to do???
# TODO: do all the broadcast operations on this change the result due to ordering?
function smvp(
    chunk::C,
    a::AbstractMatrix{Float64},
    index::CartesianIndex,
)::Float64 where {C<:Chunk}
    x, y = Tuple(index)
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
    xs, ys = haloa(chunk, hd)
    # Update left halo.
    for kk = 1:depth
        # TODO: is the +1 right?
        buffer[hd-kk+1, ys] .= buffer[hd+kk, ys]
    end
    # Update right halo.
    for kk = 1:depth
        buffer[x-hd+kk, ys] .= buffer[x-hd-1-kk, ys]
    end
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

# Copies the current u into u0
function copyu!(chunk::Chunk, hd::Int)
    H = halo(chunk, hd)
    chunk.u0[H] .= chunk.u[H]
end

# Calculates the current value of r
function residual!(chunk::Chunk, hd::Int)
    H = halo(chunk, hd)
    chunk.r[H] .= chunk.u0[H] - smvp.(chunk, Ref(chunk.u), H)
end

# Finalises the solution
function finalise!(chunk::Chunk, hd::Int)
    H = halo(chunk, hd)
    @. chunk.energy[H] = chunk.u[H] / chunk.density[H]
end

@exportAll()

end
