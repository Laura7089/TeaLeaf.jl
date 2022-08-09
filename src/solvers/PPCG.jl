module PPCG

import ..Chunk
import ..Settings

# Initialises the PPCG solver
function init(chunk::Chunk, hd::Int)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x
    @. chunk.sd[index] = chunk.r[index] / chunk.theta
end

# The PPCG inner iteration
function inner_iteration(chunk::Chunk, hd::Int, alpha::Float64, beta::Float64)
    index = (hd+1:chunk.x-hd) + (hd+1:chunk.y-hd) * chunk.x

    p = smvp(chunk.sd)
    chunk.r[index] .-= p
    chunk.u[index] .+= chunk.sd[index]

    @. chunk.sd[index] = alpha * chunk.sd[index] + beta * chunk.r[index]
end

end
