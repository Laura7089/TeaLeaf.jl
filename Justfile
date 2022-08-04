#!/usr/bin/env -S just --justfile

SRC_PATH := "./src"

# Get an interactive shell with the package imported
interactive:
    julia --project -ie 'using TeaLeaf; fa() = format("{{ SRC_PATH }}");'

# Run TeaLeaf.jl's default entrypoint
run: # No compile dep since julia handles that for us
    julia --project -e 'import TeaLeaf; TeaLeaf.main()'

# Run JuliaFormatter on the project, or a path
format path=SRC_PATH:
    julia -e 'using JuliaFormatter; format("{{ path }}")'

# Compile TeaLeaf.jl
compile:
    julia --project -e 'import TeaLeaf'

# Delete Julia TeaLeaf compilation cache
decompile:
    find ~/.julia/compiled -maxdepth 2 \
        -iname "TeaLeaf" -type d -exec rm -rfv {} \;

# Clean up run and build artefacts
clean:
    rm -fv *.tmp *.out
