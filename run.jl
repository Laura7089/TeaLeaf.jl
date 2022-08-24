#!/bin/env -S julia --project
using TeaLeaf, Logging, ArgParse, Match

s = ArgParseSettings()
@add_arg_table s begin
    "--solver", "-s"
    help = "Can be 'cg', 'cheby', 'ppcg', or 'jacobi'"
    arg_type = String
    range_tester = a -> lowercase(a) in ("cg", "cheby", "ppcg", "jacobi")
    "-x"
    help = "Number of x cells"
    arg_type = Int
    "-y"
    help = "Number of y cells"
    arg_type = Int
    "-i", "--in-file"
    help = "Settings input file"
    default = "tea.in"
    arg_type = String
    "-O", "--debug-out"
    help = "File to print debug state to"
    arg_type = String
end
args = parse_args(s)

settings = Settings(args["in-file"])
if !isnothing(args["solver"])
    @match lowercase(args["solver"]) begin
        "jacobi" => (settings.solver = TeaLeaf.Jacobi)
        "cg" => (settings.solver = TeaLeaf.CG)
        "cheby" => (settings.solver = TeaLeaf.Cheby)
        "ppcg" => (settings.solver = TeaLeaf.PPCG)
    end
end
if !isnothing(args["x"])
    settings.xcells = args["x"]
end
if !isnothing(args["y"])
    settings.ycells = args["y"]
end
if !isnothing(args["debug-out"])
    settings.debugfile = args["debug-out"]
end

chunk = initialiseapp!(settings)
@debug "Solution Parameters" settings
diffuse!(chunk, settings)
