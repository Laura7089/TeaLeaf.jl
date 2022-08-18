module PPCG

import ..Chunk
import ..Settings

# Initialises the PPCG solver
function init(chunk::Chunk, hd::Int)
    x, y = size(chunk)
    @. chunk.sd[hd+1:x-hd, hd+1:y-hd] = chunk.r[hd+1:x-hd, hd+1:y-hd] / chunk.theta
end

# The PPCG inner iteration
function inner_iteration(chunk::Chunk, hd::Int, alpha::Float64, beta::Float64)
    x, y = size(chunk)
    kk = hd+1:x-hd
    jj = hd+1:y-hd

    p = smvp.(chunk, chunk.sd, zip(kk, jj))
    chunk.r[kk, jj] .-= p
    chunk.u[kk, jj] .+= chunk.sd[kk, jj]

    @. chunk.sd[kk, jj] = alpha * chunk.sd[kk, jj] + beta * chunk.r[kk, jj]
end

end
