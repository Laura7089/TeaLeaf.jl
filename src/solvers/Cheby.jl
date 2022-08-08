module Cheby

import ..Chunk
import ..Settings

# Calculates the new value for u.
function calc_u(x::Int, y::Int, hd::Int, u::Vector{Float64}, p::Vector{Float64})
    for jj = hd+1:y-hd, kk = hd+1:x-hd
        index = kk + jj * x
        u[index] += p[index]
    end
end

# Initialises the Chebyshev solver
function init(chunk::Chunk, hd::Int)
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
        smvp = SMVP(chunk.u)
        chunk.w[index] = smvp
        chunk.r[index] = chunk.u0[index] - chunk.w[index]
        chunk.p[index] = chunk.r[index] / chunk.theta
    end

    calc_u(chunk.x, chunk.y, hd, chunk.u, chunk.p) # Done
end

# The main chebyshev iteration
function iterate(chunk::Chunk, alpha::Float64, beta::Float64)
    for jj = hd+1:chunk.y-hd, kk = hd+1:chunk.x-hd
        index = kk + jj * chunk.x
        smvp = SMVP(chunk.u)
        chunk.w[index] = smvp
        chunk.r[index] = chunk.u0[index] - chunk.w[index]
        chunk.p[index] = alpha * chunk.p[index] + beta * chunk.r[index]
    end

    calc_u(chunk.x, chunk.y, hd, chunk.u, chunk.p) # Done
end

end
