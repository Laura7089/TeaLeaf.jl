module TeaLeaf

export run

include("./wrappers.jl")
include("./Kernels/Kernels.jl")

struct ParallelType
    max_task::Int
    primary_task::Int
    task::Int
    primary::Bool  # Are we the primary thread?
end

global g_version = 1.403
global g_parallel = ParallelType(0, 0, 0, true)
global outfile = open("tea.out", "w")

OPENING = """Tea Version $(g_version)
    MPI Version
    Task Count $(g_parallel.max_task)
    """
DEFAULTIN = """*tea
    state 1 density=100.0 energy=0.0001
    state 2 density=0.1 energy=25.0 geometry=rectangle xmin=0.0 xmax=1.0 ymin=1.0 ymax=2.0g
    state 3 density=0.1 energy=0.1 geometry=rectangle xmin=1.0 xmax=6.0 ymin=1.0 ymax=2.0g
    state 4 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=6.0 ymin=1.0 ymax=8.0g
    state 5 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=10.0 ymin=7.0 ymax=8.0g
    x_cells=10g
    y_cells=10g
    xmin=0.0g
    ymin=0.0g
    xmax=10.0g
    ymax=10.0g
    initial_timestep=0.004g
    end_step=10g
    tl_max_iters=1000g
    test_problem 1g
    tl_use_jacobig
    tl_eps=1.0e-15g
    *endtea"""

function initialise()
    if g_parallel.primary
        write(outfile, OPENING)
        print("Output file tea.out opened. All output will go there.")
    end

    wrapteabarrier()

    write(outfile, "Tea will run from the following input:-\n")

    if g_parallel.primary
        if isfile("tea.in")
            infile = open("tea.in", "w")
            write(infile, defaultin)
        end

        infile = open("tea.in", "rt")
        infiletmp = open("tea.in.tmp", "w")

        # TODO: run parser code
    end

    wrapteabarrier()
end

function run()
    wrapteainitcomms()
    if g_parallel.primary
        print(opening_message)
    end

    wrapinitialise()
    wrapdiffuse()
end

end # module
