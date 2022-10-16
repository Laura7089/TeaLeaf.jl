module Kernels

using TeaLeaf
using Match
using ExportAll

const ERROR_START = 1e+10
const ERROR_SWITCH_MAX = 1.0

"""
    eigenvalues!(chunk, settings, cgiters)

Calculate the eigenvalues from `chunk.cgα` and `chunk.cβ`.

# Note

Currently not proven to be a correct reimplementation.
"""
function eigenvalues!(chunk::Chunk, settings::Settings, cgiters::Int)
    diag = 1 ./ chunk.cgα[1:cgiters]
    offdiag = zeros(cgiters)

    # Prepare matrix
    for ii = 1:cgiters
        if ii > 1
            diag[ii] += chunk.cgβ[ii-1] / chunk.cgα[ii-1]
        end
        if ii < cgiters
            offdiag[ii+1] = sqrt(chunk.cgβ[ii]) / chunk.cgα[ii]
        end
    end
    @debug "" diag offdiag

    # Calculate the eigenvalues (ignore eigenvectors)
    tqli!(diag, offdiag, cgiters)

    # TODO: can we do minmax(eigvals(SymTridiagonal(diag, offdiag))) instead?

    chunk.eigmin, chunk.eigmax = minmax(diag)

    if chunk.eigmin < 0 || chunk.eigmax < 0
        throw("Negative eigenvalue found: ($(chunk.eigmin), $(chunk.eigmax))")
    end

    # TODO: Find out the reasoning behind this!?
    # Adds some buffer for precision maybe?
    chunk.eigmin *= 0.95
    chunk.eigmax *= 1.05

    @info "Eigenvalues calculated" chunk.eigmin chunk.eigmax
end

# TODO: can we replace with inbuilt julia stuff?
"""
    tqli!(d, e, n)

Apply the TQLI algorithm to find eigenvalues of `d` and `e`.
Adapted from [this method](http://ftp.cs.stanford.edu/cs/robotics/scohen/nr/tqli.c).
"""
function tqli!(d::Vector{Float64}, e::Vector{Float64}, n::Int)
    s = r = p = g = f = dd = c = b = 0.0

    e[1:n-1] .= e[2:n]
    e[n] = 0.0

    for l = 1:n-1
        iter = 0
        while true
            m = l
            for mᵢ = l:n-2
                m = mᵢ
                dd = abs(d[mᵢ]) + abs(d[mᵢ+1])
                if e[mᵢ] == 0
                    break
                end
            end
            m == l && break

            iter += 1
            iter == 30 && throw("Too many iterations in TQLI routine")

            g = (d[l+1] - d[l]) / (2e[l])
            r = sqrt((g * g) + 1)
            g = d[m] - d[l] + e[l] / (g + (r * sign(g)))
            s = c = 1
            p = 0
            # TODO reverse iteration
            for i = m-1:l
                f = s * e[i]
                b = c * e[i]
                r = sqrt(f * f + g * g)
                e[i+1] = r
                if r == 0.0
                    d[i+1] -= p
                    e[m] = 0.0
                    continue
                end
                s = f / r
                c = g / r
                g = d[i+1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i+1] = g + p
                g = c * r - b
            end
            d[l] = d[l] - p
            e[l] = g
            e[m] = 0.0
            m != l && break
        end
    end
end

"""
    fieldsummary(chunk, settings)

Check the error margins of `chunk` according to `settings` and log the result.
"""
function fieldsummary(chunk::Chunk, set::Settings)
    !set.checkresult && return

    H = halo(chunk, set.halodepth)
    actual = sum(chunk.volume[H] .* chunk.density[H] .* chunk.u[H])
    cv = checkingvalue(set)
    @info "Checking results..." cv actual

    qa_diff = abs(100actual / cv - 100.0)
    if qa_diff < 0.001
        @info "This run PASSED" qa_diff
    else
        @warn "This run FAILED" qa_diff
    end
end

"""
    haloupdate!(chunk, set, depth[, toex][, reset])

Invoke the halo update kernels.

If `toex` is passed:
1. update those fields of `chunk`
1. call [`resettoexchange!`](@ref) on `set` if `reset == true`
1. mark them as to exchange
Else, use `set.toexchange` without modification.
"""
function haloupdate!(chunk::Chunk, set::Settings, depth::Int, toex = nothing, reset = true)
    toex = if toex != nothing
        reset && resettoexchange!(set)
        for f in toex
            set.toexchange[f] = true
        end
        toex
    else
        filter(f -> set.toexchange[f], EXCHANGE_FIELDS)
    end

    # Call `updateface` on all faces which are marked in `toex`
    updateface!.(chunk, set.halodepth, depth, getfield.(chunk, toex))
end

"""
    solvefinished!(chunk, settings)

Call all kernels on `chunk` that wrap up a solve, regardless of solver.
"""
function solvefinished!(chunk::Chunk, set::Settings)
    set.checkresult && residual!(chunk, set.halodepth)
    finalise!(chunk, set.halodepth)
    haloupdate!(chunk, set, 1, [:energy], false)
end

# TODO: what the hell is this trying to do???
"""
    smvp(chunk, chunkfield, index)

Sparse Matrix Vector Product.
"""
function smvp(chunk::Chunk, a::AbstractMatrix{Float64}, index::CartesianIndex)::Float64
    x, y = Tuple(index)
    consum = sum((1, chunk.kx[x+1, y], chunk.kx[x, y], chunk.ky[x, y+1], chunk.ky[x, y]))
    return consum * a[x, y]
    -(chunk.kx[x+1, y] * a[x+1, y] + chunk.kx[x, y] * a[x-1, y])
    -(chunk.ky[x, y+1] * a[x, y+1] + chunk.ky[x, y] * a[x, y-1])
end

"""
    updateface!(chunk, halodepth, depth, buffer)

Update faces of `buffer` in turn.
"""
function updateface!(chunk::Chunk, hd::Int, depth::Int, buffer::AbstractMatrix{Float64})
    x, y = size(chunk)
    xs, ys = haloa(chunk, hd)
    # Update left halo.
    for kk = 1:depth
        buffer[hd-kk, ys] .= buffer[hd+kk, ys]
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
        buffer[xs, hd-jj] .= buffer[xs, hd+jj]
    end
end

"""
    copyu!(chunk, halodepth)

Copy the current u into u0 within the halo produced by `halodepth`.
"""
function copyu!(chunk::Chunk, hd::Int)
    H = halo(chunk, hd)
    chunk.u0[H] .= chunk.u[H]
end

"""
    residual!(chunk, halodepth)

Update the value of `chunk.r` within `halodepth`.
"""
function residual!(chunk::Chunk, hd::Int)
    xs, ys = haloa(chunk, hd)
    for jj in ys, kk in xs
        chunk.r[kk, jj] = chunk.u0[kk, jj] - smvp(chunk, chunk.u, CartesianIndex(kk, jj))
    end
end

"""
    finalise!(chunk, halodepth)

Finalise the solution.
"""
function finalise!(chunk::Chunk, hd::Int)
    H = halo(chunk, hd)
    @. chunk.energy[H] = chunk.u[H] / chunk.density[H]
end

@exportAll()

end
