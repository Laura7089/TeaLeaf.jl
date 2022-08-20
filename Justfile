#!/usr/bin/env -S just --justfile
set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]

SRC_PATH := "src"
export JULIA_MPI_BINARY := "system"
JULIA := "julia +1.7"

# Get an interactive shell with the package imported
interactive:
    {{ JULIA }} --project -ie 'using TeaLeaf; fa() = format("{{ SRC_PATH }}");'
alias i := interactive

# Get an interactive debugger
debug:
    {{ JULIA }} --project -ie 'using Debugger, Logging, TeaLeaf; \
        disable_logging(Logging.Error)'
alias d := debug

# Run TeaLeaf.jl's default entrypoint
run: # No compile dep since julia handles that for us
    {{ JULIA }} --project -e 'using TeaLeaf; main()'
alias r := run

# Run JuliaFormatter on the project, or a path
format path=SRC_PATH:
    {{ JULIA }} -e 'using JuliaFormatter; format("{{ path }}")'
alias f := format

# Compile TeaLeaf.jl
compile:
    {{ JULIA }} --project -e 'import TeaLeaf'

# Delete Julia TeaLeaf compilation cache
decompile:
    find ~/.julia/compiled -maxdepth 2 \
        -iname "TeaLeaf" -type d -exec rm -rfv {} \;

# Clean up run and build artefacts
clean:
    rm -fv *.tmp *.out
