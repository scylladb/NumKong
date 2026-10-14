/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for NEON FHM.
 *  @file include/numkong/spatial/neonfhm.h
 *  @author Szymon Wasik
 *  @date October 14, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_neonfhm_instructions ARM NEON FP16 Widening Multiply-Add Instructions (ARMv8.4-FHM)
 *
 *      Intrinsic         Instruction                A76       M5
 *      vfmlalq_low_f16   FMLAL (V.4S, V.8H, V.8H)   4cy @ 2p  4cy @ 4p
 *      vfmlalq_high_f16  FMLAL2 (V.4S, V.8H, V.8H)  4cy @ 2p  4cy @ 4p
 *      vld1q_u16         LD1 (V.8H)                 4cy @ 2p  4cy @ 3p
 *      vaddq_f32         FADD (V.4S, V.4S, V.4S)    2cy @ 2p  2cy @ 4p
 *      vaddvq_f32        FADDP+FADDP (V.4S)         4cy @ 1p  8cy @ 1p
 *
 *  The ARMv8.4-FHM extension (FEAT_FHM) fuses the f16 → f32 widening with the multiply-add: FMLAL
 *  takes the low 4 halves of each operand and FMLAL2 the high 4, each accumulating into 4 floats.
 *  The plain NEON f16 kernels in `spatial/neon.h` spend 4 FCVTL/FCVTL2 per 8 halves before the
 *  6 FMLAs; here the 6 FMLAL/FMLAL2 are the whole inner loop, and the accumulation stays in f32.
 */
#ifndef NK_SPATIAL_NEONFHM_H
#define NK_SPATIAL_NEONFHM_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/cast/serial.h"  // `nk_partial_load_b16x8_serial_`
#include "numkong/spatial/neon.h" // `nk_angular_normalize_f32_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

NK_PUBLIC void nk_angular_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    float16x8_t a_f16x8, b_f16x8;
    // The low and high halves of every vector feed separate accumulators, so each of the three sums
    // runs as two independent chains and a core with a 4-cycle FMLAL latency is not bound by it.
    float32x4_t ab_low_f32x4 = vdupq_n_f32(0), ab_high_f32x4 = vdupq_n_f32(0);
    float32x4_t a2_low_f32x4 = vdupq_n_f32(0), a2_high_f32x4 = vdupq_n_f32(0);
    float32x4_t b2_low_f32x4 = vdupq_n_f32(0), b2_high_f32x4 = vdupq_n_f32(0);
nk_angular_f16_neonfhm_cycle:
    if (n < 8) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a, &a_vec, n);
        nk_partial_load_b16x8_serial_(b, &b_vec, n);
        a_f16x8 = vreinterpretq_f16_u16(a_vec.u16x8);
        b_f16x8 = vreinterpretq_f16_u16(b_vec.u16x8);
        n = 0;
    }
    else {
        a_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)a));
        b_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)b));
        a += 8, b += 8, n -= 8;
    }
    // FMLAL/FMLAL2: widening multiply-accumulate f16 → f32 of the elements 0-3 and 4-7
    ab_low_f32x4 = vfmlalq_low_f16(ab_low_f32x4, a_f16x8, b_f16x8);
    ab_high_f32x4 = vfmlalq_high_f16(ab_high_f32x4, a_f16x8, b_f16x8);
    a2_low_f32x4 = vfmlalq_low_f16(a2_low_f32x4, a_f16x8, a_f16x8);
    a2_high_f32x4 = vfmlalq_high_f16(a2_high_f32x4, a_f16x8, a_f16x8);
    b2_low_f32x4 = vfmlalq_low_f16(b2_low_f32x4, b_f16x8, b_f16x8);
    b2_high_f32x4 = vfmlalq_high_f16(b2_high_f32x4, b_f16x8, b_f16x8);
    if (n) goto nk_angular_f16_neonfhm_cycle;
    nk_f32_t ab = vaddvq_f32(vaddq_f32(ab_low_f32x4, ab_high_f32x4));
    nk_f32_t a2 = vaddvq_f32(vaddq_f32(a2_low_f32x4, a2_high_f32x4));
    nk_f32_t b2 = vaddvq_f32(vaddq_f32(b2_low_f32x4, b2_high_f32x4));
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM64_
#endif // NK_SPATIAL_NEONFHM_H
