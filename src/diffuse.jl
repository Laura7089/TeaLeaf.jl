using Match

# The main timestep loop
function diffuse(chunks::Vector{Chunk}, settings::Settings)
    for tt = 1:settings.end_step
        solve(chunks, settings, tt) # Done
    end
    field_summary_driver(chunks, settings, true) # Done
end

# Performs a solve for a single timestep
function solve(chunks::Vector{Chunk}, settings::Settings, tt::Int)
    # Calculate minimum timestep information
    dt = calc_min_timestep(chunks, settings.dt_init, settings.num_chunks_per_rank) # Done

    rx = dt / (settings.dx * settings.dx)
    ry = dt / (settings.dy * settings.dy)

    # Prepare halo regions for solve
    settings.fields_to_exchange .= false
    settings.fields_to_exchange[FIELD_ENERGY1] = true
    settings.fields_to_exchange[FIELD_DENSITY] = true
    halo_update!(chunks, settings, 2) # Done

    error = 1e+10

    # Perform the solve with one of the integrated solvers
    solver = @match settings.solver begin
        Jacobi => jacobi_driver # TODO
        CG => cg_driver # TODO
        Cheby => cheby_driver # TODO
        PPCG => ppcg_driver # TODO
    end
    error = solver(chunks, settings, rx, ry, error)

    # Perform solve finalisation tasks
    solve_finished_driver(chunks, settings) # Done

    if tt % settings.summary_frequency == 0
        field_summary_driver(chunks, settings, false) # Done
    end

    @info "Timestep finished" tt error
end

# Calculate minimum timestep
function calc_min_timestep(chunks::Vector{Chunk}, dt::Float64, chunks_per_task::Int)
    for cc = 2:chunks_per_task
        dtlp = chunks[cc].dt_init

        if dtlp < dt
            dt = dtlp
        end
    end
    return dt
end
