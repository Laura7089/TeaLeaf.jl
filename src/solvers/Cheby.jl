module Cheby

import ..Chunk
import ..Settings

# Initialises the Chebyshev solver
function init!(chunk::Chunk, hd::Int)
    xs, ys = haloa(chunk, hd)
    for jj in ys, kk in xs
        chunk.w[kk, jj] = smvp(chunk, chunk.u, CartesianIndex(kk, jj))
        chunk.r[kk, jj] = chunk.u0[kk, jj] - chunk.w[kk, jj]
        chunk.p[kk, jj] = chunk.r[kk, jj] / chunk.theta
    end
    chunk.u[xs, ys] .+= chunk.p[xs, ys]
end

# The main chebyshev iteration
function iterate!(chunk::Chunk, alpha::Float64, beta::Float64)
    xs, ys = haloa(chunk, hd)
    for jj in ys, kk in xs
        chunk.w[kk, jj] = smvp(chunk, chunk.u, CartesianIndex(kk, jj))
        chunk.r[kk, jj] = chunk.u0[kk, jj] - chunk.w[kk, jj]
        chunk.p[kk, jj] = alpha * chunk.p[kk, jj] + beta * chunk.r[kk, jj]
    end
    chunk.u[H] .+= chunk.p[H]
end

end
