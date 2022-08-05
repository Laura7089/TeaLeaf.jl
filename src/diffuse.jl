using Match

# The main timestep loop
function diffuse(chunks::Vector{Chunk}, settings::Settings)
    wallclock_prev = 0.0
    for tt = 2:settings.end_step
        wallclock_prev = solve(chunks, settings, tt, wallclock_prev) # Done
    end

    field_summary_driver(chunks, settings, true) # Done
end

# Performs a solve for a single timestep
function solve(chunks::Vector{Chunk}, settings::Settings, tt::Int, wallclock_prev::Float64)
    @info "Timestep $(tt)"

    # Calculate minimum timestep information
    dt = settings.dt_init
    dt = calc_min_timestep(chunks, dt, settings.num_chunks_per_rank) # Done

    rx = dt / (settings.dx * settings.dx)
    ry = dt / (settings.dy * settings.dy)

    # Prepare halo regions for solve
    settings.fields_to_exchange .= false
    settings.fields_to_exchange[FIELD_ENERGY1] = true
    settings.fields_to_exchange[FIELD_DENSITY] = true
    halo_update!(chunks, settings, 2) # Done

    error = 1e+10

    # Perform the solve with one of the integrated solvers
    error = @match settings.solver begin
        JACOBI_SOLVER => jacobi_driver(chunks, settings, rx, ry, error) # TODO
        CG_SOLVER => cg_driver(chunks, settings, rx, ry, &error) # TODO
        CHEBY_SOLVER => cheby_driver(chunks, settings, rx, ry, error) # TODO
        PPCG_SOLVER => ppcg_driver(chunks, settings, rx, ry, error) # TODO
    end

    # Perform solve finalisation tasks
    solve_finished_driver(chunks, settings) # Done

    if tt % settings.summary_frequency == 0
        field_summary_driver(chunks, settings, false) # Done
    end

    # TODO replace with @time, etc
    # double wallclock = settings->wallclock_profile->profiler_entries[0].time
    # print_and_log(settings, "Wallclock: \t\t%.3lfs\n", wallclock)
    # print_and_log(settings, "Avg. time per cell: \t%.6e\n",
    #     (wallclock-*wallclock_prev) /
    #     (settings->grid_x_cells *
    #      settings->grid_y_cells))
    @info "" error
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
