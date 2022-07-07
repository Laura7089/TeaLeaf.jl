wrapteabarrier() = ccall((:tea_barrier, "tea_leaf.so"), Nothing, ())
wrapteainitcomms() = ccall((:__tea_module_MOD_tea_init_comms, "tea_leaf.so"), Nothing, ())
wrapinitialise() = ccall((:initialise_, "tea_leaf.so"), Nothing, ())
wrapdiffuse() = ccall((:diffuse_, "tea_leaf.so"), Nothing, ())
