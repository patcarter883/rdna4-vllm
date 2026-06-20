// kernel_names.h — descriptive, type-namespaced kernel identifiers that replace the
// old arbitrary integer "version". Two scoped enums (dense vs MoE) are non-comparable,
// which structurally kills the historical collision (dense-v7 `wmma_tiled_tuned` vs
// the unrelated MoE-v7 decode GEMV).
//
// WIRE FORMAT: the torch custom op still carries an opaque `int` across the ABI — vLLM
// 0.22.69's aot_compile is fullgraph-strict and a `str` schema arg risks that capture
// path (see bindings.cpp pt2_compliant_tag). The descriptive NAMES live only above the
// torch boundary (env vars, scripts, the __init__.py wrappers map name<->int); no human
// writes the int. Enum integer values EQUAL the historical version ints so the name->int
// map and any internal numeric references stay aligned and no kernel body moves.
#pragma once

namespace w4a8 {

// Dense mmq_fp8_gemm kernels (w4a8_fp8_wmma_kernel.hip). Names from VERSIONS.md.
// (Used by the dense de-numbering step; declared here so the header is authored once.)
enum class DenseKernel : int {
    ReferenceScalar     = 0,   // scalar fp8 golden (reference only)
    RocwmmaV1           = 1,   // research ancestor of prefill_wmma
    RocwmmaTiled        = 2,   // research ancestor          (3 = retired numbering gap)
    RocwmmaPipe         = 4,   // research ancestor
    PrefillWmma         = 5,   // served: any-group-size WMMA fallback
    PrefillWmmaB128     = 6,   // research
    WmmaTiledTuned      = 7,   // research
    WmmaDbuf            = 8,   // research
    WmmaDbuf2           = 9,   // research
    PrefillWmmaAshuffle = 10,  // served: large-M prefill (A-shuffle, B-only LDS)
    DecodeGemv          = 11,  // served: decode GEMV (M<=2)
    SplitkSmallm        = 12,  // research
    RegdirectShuffle    = 13,  // research
    NsplitSmallm        = 14,  // research
};
inline bool dense_kernel_valid(int v) {
    return v == 0 || v == 1 || v == 2 || (v >= 4 && v <= 14);   // rejects the v3 gap and >14
}

// Grouped (MoE) mmq_fp8_moe_gemm kernels (moe_kernel.hip). The former v5/v6 are now ONE
// family `Wmma`: A-residence (old v5 = A-in-LDS, v6 = A-shuffle/B-only-LDS) is a
// TileConfig knob (make_moe_tile_config / VLLM_W4A8_MOE_A_IN_LDS), NOT a separate name.
enum class MoeKernel : int {
    ScalarGolden = 0,   // scalar fp8 golden (reference)
    Wmma         = 6,   // consolidated WMMA grouped GEMM (TileConfig-driven; served default)
    Gemv         = 7,   // decode GEMV (scalar-dot; outside the MMA policy)
};
inline bool moe_kernel_valid(int v) {
    return v == 0 || v == 6 || v == 7;
}

}  // namespace w4a8
