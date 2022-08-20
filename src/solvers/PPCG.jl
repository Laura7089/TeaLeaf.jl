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
    H = halo(chunk, hd)
    chunk.r[H] .-= smvp.(chunk, Ref(chunk.sd), H)
    chunk.u[H] .+= chunk.sd[H]
    @. chunk.sd[H] = alpha * chunk.sd[H] + beta * chunk.r[H]
end

end
