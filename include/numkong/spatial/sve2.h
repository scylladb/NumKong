/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for SVE2.
 *  @file include/numkong/spatial/sve2.h
 *  @author Szymon Wasik
 *  @date October 14, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_sve2_instructions ARM SVE2 Widening Multiply-Add Instructions
 *
 *      Intrinsic      Instruction                V2
 *      svld1_f16      LD1H (Z.H, P/Z, [Xn])      6cy @ 3p
 *      svmlalb_f32    FMLALB (Z.S, Z.H, Z.H)     4cy @ 4p
 *      svmlalt_f32    FMLALT (Z.S, Z.H, Z.H)     4cy @ 4p
 *      svadd_f32_x    FADD (Z.S, Z.S, Z.S)       2cy @ 4p
 *      svaddv_f32     FADDV (S, P, Z.S)          6cy @ 1p
 *      svwhilelt_b16  WHILELT (P.H, Xn, Xm)      2cy @ 1p
 *      svcnth         CNTH (Xd)                  1cy @ 2p
 *
 *  FMLALB and FMLALT are part of the base SVE2 instruction set (FEAT_SVE2, as on Neoverse N2 and V2,
 *  hence AWS Graviton4): they widen the bottom (even) or top (odd) f16 lane of each 32-bit lane pair
 *  to f32 and multiply-add it into an f32 accumulator, with no predicate. The f16 kernels in
 *  `spatial/svehalf.h` only have SVE + FP16, so for every vector of halves they issue 4 FCVT, 2 EXT
 *  (to bring the odd lanes to even positions), 3 WHILELT and 6 merging FMLAs; here the 6
 *  FMLALB/FMLALT are the whole inner loop, and the accumulation stays in f32.
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  use 128-bit. Code using svcnth() adapts automatically, but wider vectors process more
 *  elements per iteration with identical latencies.
 */
#ifndef NK_SPATIAL_SVE2_H
#define NK_SPATIAL_SVE2_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVE2

#include "numkong/types.h"
#include "numkong/reduce/sve.h"   // `nk_svaddv_f32_`
#include "numkong/spatial/neon.h" // `nk_angular_normalize_f32_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+sve2")
#endif

NK_PUBLIC void nk_angular_f16_sve2(nk_f16_t const *a_enum, nk_f16_t const *b_enum, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    // The bottom and top f16 lanes feed separate accumulators, so each of the three sums runs as
    // two independent chains and a core with a 4-cycle FMLALB/FMLALT latency is not bound by it.
    svfloat32_t ab_bot_f32x = svdup_n_f32(0.0f), ab_top_f32x = svdup_n_f32(0.0f);
    svfloat32_t a2_bot_f32x = svdup_n_f32(0.0f), a2_top_f32x = svdup_n_f32(0.0f);
    svfloat32_t b2_bot_f32x = svdup_n_f32(0.0f), b2_top_f32x = svdup_n_f32(0.0f);
    nk_f16_for_arm_simd_t const *a = (nk_f16_for_arm_simd_t const *)(a_enum);
    nk_f16_for_arm_simd_t const *b = (nk_f16_for_arm_simd_t const *)(b_enum);
    do {
        // The zeroing loads leave the lanes past `n` at zero, which add nothing to the sums,
        // so the multiply-adds need no predicate of their own.
        svbool_t predicate_b16x = svwhilelt_b16_u64(i, n);
        svfloat16_t a_f16x = svld1_f16(predicate_b16x, a + i);
        svfloat16_t b_f16x = svld1_f16(predicate_b16x, b + i);
        ab_bot_f32x = svmlalb_f32(ab_bot_f32x, a_f16x, b_f16x);
        ab_top_f32x = svmlalt_f32(ab_top_f32x, a_f16x, b_f16x);
        a2_bot_f32x = svmlalb_f32(a2_bot_f32x, a_f16x, a_f16x);
        a2_top_f32x = svmlalt_f32(a2_top_f32x, a_f16x, a_f16x);
        b2_bot_f32x = svmlalb_f32(b2_bot_f32x, b_f16x, b_f16x);
        b2_top_f32x = svmlalt_f32(b2_top_f32x, b_f16x, b_f16x);
        i += svcnth();
    } while (i < n);

    svbool_t all_b32x = svptrue_b32();
    nk_f32_t ab_f32 = nk_svaddv_f32_(all_b32x, svadd_f32_x(all_b32x, ab_bot_f32x, ab_top_f32x));
    nk_f32_t a2_f32 = nk_svaddv_f32_(all_b32x, svadd_f32_x(all_b32x, a2_bot_f32x, a2_top_f32x));
    nk_f32_t b2_f32 = nk_svaddv_f32_(all_b32x, svadd_f32_x(all_b32x, b2_bot_f32x, b2_top_f32x));
    *result = nk_angular_normalize_f32_neon_(ab_f32, a2_f32, b2_f32);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SVE2
#endif // NK_TARGET_ARM64_
#endif // NK_SPATIAL_SVE2_H
