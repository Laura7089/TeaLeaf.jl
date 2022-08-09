using Match

const ERROR_START = 1e+10

# The main timestep loop
function diffuse(chunks::Vector{Chunk}, settings::Settings)
    for tt = 1:settings.end_step
        # Calculate minimum timestep information
        dt = calc_min_timestep(chunks, settings.dt_init, settings.num_chunks_per_rank) # Done

        rx = dt / (settings.dx * settings.dx)
        ry = dt / (settings.dy * settings.dy)

        # Prepare halo regions for solve
        settings.fields_to_exchange .= false
        settings.fields_to_exchange[FIELD_ENERGY1] = true
        settings.fields_to_exchange[FIELD_DENSITY] = true
        # TODO: is depth=1 correct here?
        halo_update!(chunks, settings, 1) # Done

        # Perform the solve with one of the integrated solvers
        error = settings.solver.driver(chunks, settings, rx, ry)

        # Perform solve finalisation tasks
        solve_finished_driver(chunks, settings) # Done

        if tt % settings.summary_frequency == 0
            field_summary(chunks, settings, false) # Done
        end

        @info "Timestep finished" tt error
    end
    field_summary(chunks, settings, true) # Done
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
