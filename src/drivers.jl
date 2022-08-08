include("./solvers/CG.jl")
include("./solvers/Jacobi.jl")
include("./solvers/Cheby.jl")
include("./solvers/PPCG.jl")

function field_summary(chunks::Vector{Chunk}, settings::Settings, is_solve_finished::Bool)
    vol = 0.0
    ie = 0.0
    temp = 0.0
    mass = 0.0

    for c in chunks
        for jj = settings.halo_depth+1:c.y-settings.halo_depth,
            kk = settings.halo_depth+1:c.x-settings.halo_depth

            index = kk + jj * c.x
            cellVol = c.volume[index]
            cellMass = cellVol * c.density[index]
            vol += cellVol
            mass += cellMass
            ie += cellMass * c.energy0[index]
            temp += cellMass * c.u[index]
        end
    end

    if settings.check_result && is_solve_finished
        @info "Checking results..."

        checking_value = get_checking_value(settings) # Done

        @info "Expected and actual:" checking_value temp

        qa_diff = abs(100.0 * (temp / checking_value) - 100.0)
        if qa_diff < 0.001
            @info "This run PASSED" qa_diff
        else
            @warn "This run FAILED" qa_diff
        end
    end
end

# Invoke the halo update kernels
function halo_update!(chunks::Vector{Chunk}, settings::Settings, depth::Int)
    # Check that we actually have exchanges to perform
    if !any(settings.fields_to_exchange)
        return
    end

    for c in chunks
        for (index, buffer) in [
            (FIELD_DENSITY, :density),
            (FIELD_P, :p),
            (FIELD_ENERGY0, :energy0),
            (FIELD_ENERGY1, :energy),
            (FIELD_U, :u),
            (FIELD_SD, :sd),
        ]
            if settings.fields_to_exchange[index]
                update_face!(c, settings.halo_depth, getfield(c, buffer))
            end
        end
    end
end

# Calls all kernels that wrap up a solve regardless of solver
function solve_finished_driver(chunks::Vector{Chunk}, settings::Settings)
    exact_error = 0.0

    if settings.check_result
        for c in chunks
            calculate_residual(c, settings.halo_depth) # Done

            exact_error += calculate_2norm(c, settings.halo_depth, c.r) # Done
        end
    end

    finalise.(chunks, settings.halo_depth) # Done
    settings.fields_to_exchange[FIELD_ENERGY1] = true
    halo_update!(chunks, settings, 1) # Done
end
