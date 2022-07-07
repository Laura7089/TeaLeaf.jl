wrapteablockinit(
    x_min,
    x_max,
    y_min,
    y_max,
    halo_exchange_depth,
    cp,
    bfp,
    Kx,
    Ky,
    Di,
    rx,
    ry,
) = cccall(
    (:tea_leaf_common_kernel_module_MOD_tea_block_init, "tea_leaf"),
    Nothing,
    (
        Int,
        Int,
        Int,
        Int,
        Int,
        Matrix{Float},
        Matrix{Float},
        Matrix{Float},
        Matrix{Float},
        Matrix{Float},
        Float,
        Float,
    ),
    (x_min, x_max, y_min, y_max, halo_exchange_depth, cp, bfp, Kx, Ky, Di, rx, ry),
)

wrapteadiaginit(x_min, x_max, y_min, y_max, halo_exchange_depth, Mi, Kx, Ky, Di, rx, ry) =
    ccall(
        (:tea_leaf_common_kernel_module_MOD_tea_diag_init, "tea_leaf"),
        Nothing,
        (
            Int,
            Int,
            Int,
            Int,
            Int,
            Matrix{Float},
            Matrix{Float},
            Matrix{Float},
            Matrix{Float},
            Float,
            Float,
        ),
        (x_min, x_max, y_min, y_max, halo_exchange_depth, Mi, Kx, Ky, Di, rx, ry),
    )