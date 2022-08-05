function initialise_application()
    settings = parse_flags()
    states = read_config!(settings) # Done
    settings.num_chunks = settings.num_ranks * settings.num_chunks_per_rank

    chunks = decompose_field!(settings) # Done

    for c in chunks
        set_chunk_data!(settings, c) # Done
    end

    for c in chunks
        set_chunk_state!(c, states) # Done
    end

    # Prime the initial halo data
    settings.fields_to_exchange .= false
    settings.fields_to_exchange[FIELD_DENSITY] = true
    settings.fields_to_exchange[FIELD_ENERGY0] = true
    settings.fields_to_exchange[FIELD_ENERGY1] = true
    halo_update!(chunks, settings, 2) # Done

    for c in chunks
        store_energy(c) # Done
    end

    return (settings, chunks)
end

# Decomposes the field into multiple chunks
function decompose_field!(settings::Settings)::Vector{Chunk}
    chunks = Array{Chunk}(undef, settings.num_chunks)

    # Calculates the chunks field is to be decomposed into
    x_chunks, y_chunks = begin
        xc = settings.grid_x_cells
        yc = settings.grid_y_cells
        nc = settings.num_chunks
        allposs = map(x -> (x, div(nc, x)), 1:nc)

        _, i = findmin(allposs) do (x, y)
            perim = (xc / x)^2 + (yc / y)^2
            area = (xc / x) * (yc / y)
            perim / area
        end
        allposs[i]
    end

    @info "Chose decomposition:" (x_chunks, y_chunks)

    dx, mod_x = divrem(settings.grid_x_cells, x_chunks)
    dy, mod_y = divrem(settings.grid_y_cells, y_chunks)

    add_x_prev = 0
    add_y_prev = 0

    # Compute the full decomposition on all ranks
    for yy = 0:y_chunks-1
        add_y = yy < mod_y

        for xx = 1:x_chunks
            add_x = xx < mod_x
            # TODO: this will first evaluate to 2 at the moment
            cc = xx + yy * x_chunks

            # Set up the mesh ranges
            left = xx * dx + add_x_prev
            bottom = yy * dy + add_y_prev

            chunks[cc] = Chunk(settings, dx + add_x, dy + add_y, left, bottom)

            # Set up the chunk connectivity
            chunks[cc].neighbours[CHUNK_LEFT] = (xx == 0) ? EXTERNAL_FACE : cc - 1
            chunks[cc].neighbours[CHUNK_RIGHT] =
                (xx == x_chunks) ? EXTERNAL_FACE : cc + 1
            chunks[cc].neighbours[CHUNK_BOTTOM] =
                (yy == 0) ? EXTERNAL_FACE : cc - x_chunks
            chunks[cc].neighbours[CHUNK_TOP] =
                (yy == y_chunks) ? EXTERNAL_FACE : cc + x_chunks

            # If chunks rounded up, maintain relative location
            add_x_prev += add_x
        end
        add_x_prev = 0
        add_y_prev += add_y
    end

    return chunks
end
