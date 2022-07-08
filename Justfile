#!/usr/bin/env -S just --justfile

REF_DIR := "./src/ref"

# Get an interactive shell with the package imported
interactive:
    julia -i <(echo 'using Revise; using Pkg; Pkg.activate("."); using TeaLeaf')

# Run TeaLeaf.jl's default entrypoint
run: # No compile dep since julia handles that for us
    julia <(echo 'import Pkg; Pkg.activate("."); import TeaLeaf; TeaLeaf.run()')

# Compile TeaLeaf.jl
compile:
    julia <(echo 'import Pkg; Pkg.activate("."); import TeaLeaf')

# Compile tealeaf fortran reference
compile_ref:
    cd "{{ REF_DIR }}" && \
        make COMPILER=GNU MPI_COMPILER=mpifort C_MPI_COMPILER=mpicc shared

# Delete Julia TeaLeaf compilation cache
decompile:
    find ~/.julia/compiled -maxdepth 2 \
        -iname "TeaLeaf" -type d -exec rm -rfv {} \;

# Print a list of functions available in the reference
ref_funcs: compile_ref
    @readelf -sW "{{ REF_DIR }}/tea_leaf.so" | grep FUNC

# Clean up run and build artefacts
clean:
    rm -fv *.tmp *.out ./deps/build.log
    cd "{{ REF_DIR }}" && make clean
