# TeaLeaf.jl

This is a port of [TeaLeaf](https://github.com/UK-MAC/TeaLeaf) to the [Julia Programming Language](https://julialang.org/).
It was ported from [this C implementation](https://github.com/UoB-HPC/TeaLeaf) of TeaLeaf.
See [the issues tab](https://github.com/Laura7089/TeaLeaf.jl/issues) for an idea of the status of the project.

## Usage

The code is intended to use the same configuration file format as other TeaLeaf implementations; an example configuration fileset (`tea.in`, `tea.problems`) is provided.

### Running

#### Using `just`

Using [`just`](https://github.com/casey/just) is the recommended way to run the program.

`just run` will run the program with the default parameters.
To see more options, see the output of `just -l`.

#### Manually

There are a handful of options for running the program manually:

- It can be run straight from this directory using the quick-run script with `julia --project ./run.jl`.
  Extra CLI args can be appended with `-- --extra-arg`.
  See `julia --project ./run.jl -- --help` for options.
- It can be run from the Julia REPL started with `julia --project` (see the documentation for this).

### Documentation

## Licensing

All original code in this repo is licensed under the Apache license.
