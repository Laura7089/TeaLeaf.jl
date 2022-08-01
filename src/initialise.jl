function initialise_application(chunks::Vector{Chunk}, settings::Settings)
    states::Vector{State}
    read_config(settings, states)

    chunks = Array{Chunk}(undef, settings.num_chunks_per_rank)

    decompose_field(settings, chunks)
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
