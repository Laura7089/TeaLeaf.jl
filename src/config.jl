function read_config(Settings* settings, State** states)
    # Open the configuration file
    open(settings.tea_in_filename, read = true) do
        # Read all of the settings from the config
        read_settings(tea_in, settings)

        # Read in the states
        settings.num_states = read_states(tea_in, settings, states)
    end

    initialise_log(settings)

    @info "Solution Parameters" settings

    for ss in 0:settings.num_states
        @info "state ${ss}", ss states[ss].density states[ss].energy
        if ss > 0
            @info "" states[ss].x_min states[ss].y_min states[ss].x_max states[ss].y_max states[ss].radius states[ss].geometry
        end
    end
end
