@debug("Building TeaLeaf reference implementation...")
cd("fort_ref")
run(`make COMPILER=GNU C_MPI_COMPILER=mpicc MPI_COMPILER=mpifort shared`)
