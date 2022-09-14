#!/usr/bin/env -S just --justfile
set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]
set positional-arguments
set dotenv-load

SRC_PATH := "./src"
RUN_FILE := "./run.jl"
export JULIA_MPI_BINARY := "system"
JULIA := env_var_or_default("JULIA", "julia")

# Get an interactive shell with the package imported
interactive:
    {{ JULIA }} --project -ie 'using TeaLeaf; fa() = format("{{ SRC_PATH }}");'
alias i := interactive

# Get an interactive debugger
debug:
    @# We disable logging because Debugger.jl doesn't work with it at
    @# the time of writing.
    @# See https://github.com/JuliaDebug/Debugger.jl/issues/318
    {{ JULIA }} --project -ie 'using Debugger, Logging, TeaLeaf; \
        disable_logging(Logging.Error)'
alias d := debug

# Run TeaLeaf.jl's default entrypoint
run *args="":
    {{ JULIA }} --project "{{ RUN_FILE }}" -- {{ args }}
alias r := run

# Run JuliaFormatter on the project, or a path
format path=SRC_PATH:
    {{ JULIA }} -e 'using JuliaFormatter; format("{{ path }}")'
alias f := format
