module PPCG

import ..Chunk
import ..Settings

# Initialises the PPCG solver
function init(chunk::Chunk, hd::Int)
    H = halo(chunk, hd)
    @. chunk.sd[H] = chunk.r[H] / chunk.theta
end

# The PPCG inner iteration
function inneriteration!(chunk::Chunk, hd::Int, alpha::Float64, beta::Float64)
    xs, ys = haloa(chunk, hd)
    for jj in ys, kk in xs
        chunk.r[kk, jj] -= smvp(chunk, chunk.sd, CartesianIndex(kk, jj))
        chunk.u[kk, jj] += chunk.sd[kk, jj]
        chunk.sd[kk, jj] = alpha * chunk.sd[kk, jj] + beta * chunk.r[kk, jj]
    end
end

end
