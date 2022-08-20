module Cheby

import ..Chunk
import ..Settings

# Initialises the Chebyshev solver
function init!(chunk::Chunk, hd::Int)
    H = halo(chunk, hd)
    chunk.w[H] .= smvp.(chunk, Ref(chunk.u), H)
    chunk.r[H] .= chunk.u0[H] .- chunk.w[H]
    chunk.p[H] .= chunk.r[H] ./ chunk.theta
    chunk.u[H] .+= chunk.p[H]
end

# The main chebyshev iteration
function iterate!(chunk::Chunk, alpha::Float64, beta::Float64)
    H = halo(chunk, hd)
    chunk.w[H] .= smvp.(chunk, Ref(chunk.u), H)
    chunk.r[H] .= chunk.u0[H] .- chunk.w[H]
    @. chunk.p[H] = alpha * chunk.p[H] + beta * chunk.r[H]
    chunk.u[H] .+= chunk.p[H]
end

end
