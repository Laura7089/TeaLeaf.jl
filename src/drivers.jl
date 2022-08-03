# Invokes the set chunk data kernel
function field_summary_driver(
    chunks::Vector{Chunk},
    settings::Settings,
    is_solve_finished::Bool,
)
    vol = 0.0
    ie = 0.0
    temp = 0.0
    mass = 0.0

    for cc = 2:settings.num_chunks_per_rank
        vol, ie, temp, mass = field_summary(chunks[cc], settings, vol, mass, ie, temp) # Done
    end

    if settings.check_result && is_solve_finished
        @info "Checking results..."

        checking_value = 1.0
        checking_value = get_checking_value(settings, checking_value) # TODO

        @info "Expected and actual:" checking_value temp

        qa_diff = abs(100.0 * (temp / checking_value) - 100.0)
        if qa_diff < 0.001
            @info "This run PASSED"
        else
            @error "This run FAILED"
        end
        @info " Difference is within:" qa_diff
    end
end

# Performs a full solve with the CG solver kernels
function cg_driver(
    chunks::Vector{Chunk},
    settings::Settings,
    rx::Float64,
    ry::Float64,
    error::Float64,
) # TODO: modifies `error`
    tt::Int

    # Perform CG initialisation
    rro = cg_init_driver(chunks, settings, rx, ry, rro) # Done

    # Iterate till convergence
    for tt = 2:settings.max_iters
        rro, error = cg_main_step_driver(chunks, settings, tt, rro, error) # Done

        halo_update_driver(chunks, settings, 1) # Done

        if sqrt(abs(error)) < settings.eps
            break
        end
    end

    @info "Iterations" tt
    return error
end

# Invokes the CG initialisation kernels
function cg_init_driver(
    chunks::Vector{Chunk},
    settings::Settings,
    rx::Float64,
    ry::Float64,
    rro::Float64,
) # TODO: modifies `rro`
    rro = 0.0

    for cc = 2:settings.num_chunks_per_rank
        cg_init(chunks[cc], settings.halo_depth, settings.coefficient, rx, ry, rro) # Done
    end

    # Need to update for the matvec
    reset_fields_to_exchange(settings) # Done
    settings.fields_to_exchange[FIELD_U] = true
    settings.fields_to_exchange[FIELD_P] = true
    halo_update_driver(chunks, settings, 1) # Done

    for cc = 2:settings.num_chunks_per_rank
        copy_u(chunks[cc], settings.halo_depth) # Done
    end

    return rro
end

# Invokes the main CG solve kernels
function cg_main_step_driver(
    chunks::Vector{Chunk},
    settings::Settings,
    tt::Int,
    rro::Float64,
    error::Float64,
) # TODO: modifies error and rro
    pw = 0.0

    for cc = 2:settings.num_chunks_per_rank
        pw = cg_calc_w(chunks[cc], settings.halo_depth, pw) # Done
    end

    alpha = rro / pw
    rrn = 0.0

    for cc = 2:settings.num_chunks_per_rank
        # TODO: Some redundancy across chunks??
        chunks[cc].cg_alphas[tt] = alpha

        rrn = cg_calc_ur(chunks[cc], settings.halo_depth, alpha, rrn) # Done
    end

    beta = rrn / rro

    for cc = 2:settings.num_chunks_per_rank
        # TODO: Some redundancy across chunks??
        chunks[cc].cg_betas[tt] = beta

        cg_calc_p(chunks[cc], settings.halo_depth, beta) # Done
    end

    return (rrn, rrn)
end

# Invoke the halo update kernels
function halo_update_driver(chunks::Vector{Chunk}, settings::Settings, depth::Int)
    # Check that we actually have exchanges to perform
    if !any(settings.fields_to_exchange)
        return
    end

    # Looks like this is a no-op in the original with MPI disabled
    # remote_halo_driver(chunks, settings, depth)

    for cc = 2:settings.num_chunks_per_rank
        local_halos!(chunks[cc], settings) # Done
    end
end

# Calls all kernels that wrap up a solve regardless of solver
function solve_finished_driver(chunks::Vector{Chunk}, settings::Settings)
    exact_error = 0.0

    if settings.check_result
        for cc in 0+1:settings.num_chunks_per_rank
            calculate_residual(chunks[cc], settings.halo_depth) # Done

            exact_error = calculate_2norm( chunks[cc], settings.halo_depth, chunks[cc].r, exact_error) # Done
        end
    end

    for cc in 0+1:settings.num_chunks_per_rank
        finalise(chunks[cc], settings.halo_depth) # Done
    end

    settings.fields_to_exchange[FIELD_ENERGY1] = true
    halo_update_driver(chunks, settings, 1) # Done
end
