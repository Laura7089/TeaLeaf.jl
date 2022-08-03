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
    rro = cg_init_driver(chunks, settings, rx, ry, rro) # Done

    # Iterate till convergence
    for tt = 2:settings.max_iters
        rro, error = cg_main_step_driver(chunks, settings, tt, rro, error) # Done

        halo_update_driver(chunks, settings, 1) # TODO

        if sqrt(abs(error)) < settings.eps
            break
        end
    end

    @info "Iterations" tt
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
        cg_init(chunks[cc], settings.halo_depth, settings.coefficient, rx, ry, rro) # Done
    end

    # Need to update for the matvec
    reset_fields_to_exchange(settings) # TODO
    settings.fields_to_exchange[FIELD_U] = true
    settings.fields_to_exchange[FIELD_P] = true
    halo_update_driver(chunks, settings, 1) # TODO

    sum_over_ranks(settings, rro) # TODO

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

    pw = sum_over_ranks(settings, pw) # TODO

    alpha = rro / pw
    rrn = 0.0

    for cc = 2:settings.num_chunks_per_rank
        # TODO: Some redundancy across chunks??
        chunks[cc].cg_alphas[tt] = alpha

        rrn = cg_calc_ur(chunks[cc], settings.halo_depth, alpha, rrn) # Done
    end

    rrn = sum_over_ranks(settings, rrn) # TODO

    beta = rrn / rro

    for cc = 2:settings.num_chunks_per_rank
        # TODO: Some redundancy across chunks??
        chunks[cc].cg_betas[tt] = beta

        cg_calc_p(chunks[cc], settings.halo_depth, beta) # Done
    end

    return (rrn, rrn)
end
