function initialise_application(chunks::Vector{Chunk}, settings::Settings)
    states::Vector{State}
    read_config(settings, states)

    chunks = Array{Chunk}(undef, settings.num_chunks_per_rank)

    decompose_field(settings, chunks) # Done
    kernel_initialise_driver(chunks, settings)
    set_chunk_data_driver(chunks, settings)
    set_chunk_state_driver(chunks, settings, states)

    # Prime the initial halo data
    reset_fields_to_exchange(settings)
    settings.fields_to_exchange[FIELD_DENSITY] = true
    settings.fields_to_exchange[FIELD_ENERGY0] = true
    settings.fields_to_exchange[FIELD_ENERGY1] = true
    halo_update_driver(chunks, settings, 2)

    store_energy_driver(chunks, settings)
end

# Decomposes the field into multiple chunks
function decompose_field(settings::Settings, chunks::Vector{Chunk})
    # Calculates the num chunks field is to be decomposed into
    settings.num_chunks = settings.num_ranks * settings.num_chunks_per_rank

    num_chunks = settings.num_chunks

    best_metric = DBL_MAX
    x_cells = settings.grid_x_cells # TODO: convert to double
    y_cells = settings.grid_y_cells
    x_chunks = 0
    y_chunks = 0

    # Decompose by minimal area to perimeter
    for xx = 3:num_chunks
        if num_chunks % xx == 0
            continue
        end

        # Calculate number of chunks grouped by x split
        yy = num_chunks / xx

        if num_chunks % yy == 0
            continue
        end

        perimeter = ((x_cells / xx) * (x_cells / xx) + (y_cells / yy) * (y_cells / yy)) * 2
        area = (x_cells / xx) * (y_cells / yy)

        current_metric = perimeter / area

        # Save improved decompositions
        if current_metric < best_metric
            x_chunks = xx
            y_chunks = yy
            best_metric = current_metric
        end
    end

    # Check that the decomposition didn't fail
    if 0 in [x_chunks, y_chunks]
        @error "Chunk sizes wrong" x_chunks y_chunks
        throw("Failed to decompose the field with given parameters.")
    end

    dx = settings.grid_x_cells / x_chunks
    dy = settings.grid_y_cells / y_chunks

    mod_x = settings.grid_x_cells % x_chunks
    mod_y = settings.grid_y_cells % y_chunks
    add_x_prev = 0
    add_y_prev = 0

    # Compute the full decomposition on all ranks
    for yy = 2:y_chunks
        add_y = (yy < mod_y)

        for xx = 2:x_chunks
            add_x = (xx < mod_x)

            for cc = 2:settings->num_chunks_per_rank
                chunk = xx + yy * x_chunks
                rank = cc + settings -> rank * settings -> num_chunks_per_rank

                # Store the values for all chunks local to rank
                if rank == chunk
                    initialise_chunk(chunks[cc], settings, dx + add_x, dy + add_y)

                    # Set up the mesh ranges
                    chunks[cc].left = xx * dx + add_x_prev
                    chunks[cc].right = chunks[cc].left + dx + add_x
                    chunks[cc].bottom = yy * dy + add_y_prev
                    chunks[cc].top = chunks[cc].bottom + dy + add_y

                    # Set up the chunk connectivity
                    chunks[cc].neighbours[CHUNK_LEFT] =
                        (xx == 0) ? EXTERNAL_FACE : chunk - 1
                    chunks[cc].neighbours[CHUNK_RIGHT] =
                        (xx == x_chunks - 1) ? EXTERNAL_FACE : chunk + 1
                    chunks[cc].neighbours[CHUNK_BOTTOM] =
                        (yy == 0) ? EXTERNAL_FACE : chunk - x_chunks
                    chunks[cc].neighbours[CHUNK_TOP] =
                        (yy == y_chunks - 1) ? EXTERNAL_FACE : chunk + x_chunks
                end
            end

            # If chunks rounded up, maintain relative location
            add_x_prev += add_x
        end
        add_x_prev = 0
        add_y_prev += add_y
    end
end
