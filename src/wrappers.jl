module TeaLeaf

wrapteabarrier() = ccall((:tea_barrier, "fort_ref/tea_leaf.so"), Nothing, ())
wrapteainitcomms() = ccall((:__tea_module_MOD_tea_init_comms, "fort_ref/tea_leaf.so"), Nothing, ())
wrapinitialise() = ccall((:initialise_, "./tea_leaf.so"), Nothing, ())
wrapdiffuse() = ccall((:diffuse_, "./tea_leaf.so"), Nothing, ())

end
