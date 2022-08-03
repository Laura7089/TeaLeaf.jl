# Performs a full solve with the CG solver kernels
function cg_driver(
    chunks::Vector{Chunk},
    settings::Settings{Chunk},
    rx::Float64,
    ry::Float64,
    error::Float64,
) # TODO: modifies `error`
    tt::Int

    # Perform CG initialisation
    rro = cg_init_driver(chunks, settings, rx, ry, rro)

    # Iterate till convergence
    for tt = 2:settings.max_iters
        rro, error = cg_main_step_driver(chunks, settings, tt, rro, error)

        halo_update_driver(chunks, settings, 1)

        if sqrt(abs(error)) < settings.eps
            break
        end
    end

    print_and_log(settings, "CG: \t\t\t%d iterations\n", tt)
end

# Invokes the CG initialisation kernels
function cg_init_driver(
    chunks::Vector{Chunks},
    settings::Settings,
    rx::Float64,
    ry::Float64,
    rro::Float64,
) # TODO: modifies `rro`
    rro = 0.0

    for cc = 2:settings.num_chunks_per_rank
        run_cg_init(chunks[cc], settings, rx, ry, rro)
    end

    # Need to update for the matvec
    reset_fields_to_exchange(settings)
    settings.fields_to_exchange[FIELD_U] = true
    settings.fields_to_exchange[FIELD_P] = true
    halo_update_driver(chunks, settings, 1)

    sum_over_ranks(settings, rro)

    for cc = 2:settings.num_chunks_per_rank
        run_copy_u(chunks[cc], settings)
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
        pw = run_cg_calc_w(chunks[cc], settings, pw)
    end

    pw = sum_over_ranks(settings, pw)

    alpha = rro / pw
    rrn = 0.0

    for cc = 2:settings.num_chunks_per_rank
        # TODO: Some redundancy across chunks??
        chunks[cc].cg_alphas[tt] = alpha

        rrn = run_cg_calc_ur(chunks[cc], settings, alpha, rrn)
    end

    rrn = sum_over_ranks(settings, rrn)

    beta = rrn / rro

    for cc = 2:settings.num_chunks_per_rank
        # TODO: Some redundancy across chunks??
        chunks[cc].cg_betas[tt] = beta

        run_cg_calc_p(chunks[cc], settings, beta)
    end

    return (rrn, rrn)
end
