/*
 * This file is derivative of part of the SoftFloat IEC/IEEE
 * Floating-point Arithmetic Package, Release 2b.
*/

/*============================================================================

This C header file is part of the SoftFloat IEC/IEEE Floating-point Arithmetic
Package, Release 2b.

Written by John R. Hauser.  This work was made possible in part by the
International Computer Science Institute, located at Suite 600, 1947 Center
Street, Berkeley, California 94704.  Funding was partially provided by the
National Science Foundation under grant MIP-9311980.  The original version
of this code was written as part of a project to build a fixed-point vector
processor in collaboration with the University of California at Berkeley,
overseen by Profs. Nelson Morgan and John Wawrzynek.  More information
is available through the Web page `http://www.cs.berkeley.edu/~jhauser/
arithmetic/SoftFloat.html'.

THIS SOFTWARE IS DISTRIBUTED AS IS, FOR FREE.  Although reasonable effort has
been made to avoid it, THIS SOFTWARE MAY CONTAIN FAULTS THAT WILL AT TIMES
RESULT IN INCORRECT BEHAVIOR.  USE OF THIS SOFTWARE IS RESTRICTED TO PERSONS
AND ORGANIZATIONS WHO CAN AND WILL TAKE FULL RESPONSIBILITY FOR ALL LOSSES,
COSTS, OR OTHER PROBLEMS THEY INCUR DUE TO THE SOFTWARE, AND WHO FURTHERMORE
EFFECTIVELY INDEMNIFY JOHN HAUSER AND THE INTERNATIONAL COMPUTER SCIENCE
INSTITUTE (possibly via similar legal warning) AGAINST ALL LOSSES, COSTS, OR
OTHER PROBLEMS INCURRED BY THEIR CUSTOMERS AND CLIENTS DUE TO THE SOFTWARE.

Derivative works are acceptable, even for commercial purposes, so long as
(1) the source code for the derivative work includes prominent notice that
the work is derivative, and (2) the source code includes prominent notice with
these four paragraphs for those parts of this code that are retained.

=============================================================================*/

#ifndef SHAREMIND_SOFTFLOAT_SOFTFLOAT_H
#define SHAREMIND_SOFTFLOAT_SOFTFLOAT_H

#include "milieu.h"


#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(internal)

/*----------------------------------------------------------------------------
| Software IEC/IEEE floating-point types.
*----------------------------------------------------------------------------*/
typedef sf_bits32 sf_float32;
typedef sf_bits64 sf_float64;

/*----------------------------------------------------------------------------
| Software IEC/IEEE floating-point underflow tininess-detection mode.
| bit 0
*----------------------------------------------------------------------------*/
#define sf_fpu_state_tininess_mask (0x01)
#define sf_float_tininess_after_rounding  (0x0)
#define sf_float_tininess_before_rounding (0x1)

/*----------------------------------------------------------------------------
| Software IEC/IEEE floating-point rounding mode.
| bits 1 and 2
*----------------------------------------------------------------------------*/
#define sf_fpu_state_rounding_mask (0x06)
#define sf_float_round_nearest_even (0x00 << 1u)
#define sf_float_round_to_zero      (0x01 << 1u)
#define sf_float_round_down         (0x02 << 1u)
#define sf_float_round_up           (0x03 << 1u)

/*----------------------------------------------------------------------------
| Software IEC/IEEE floating-point exception flags as crash values.
| bits 3 to 8
*----------------------------------------------------------------------------*/
#define sf_fpu_state_exception_crash_mask (0xf8)
#define sf_float_flag_crash_inexact   (0x01 << 3u)
#define sf_float_flag_crash_underflow (0x02 << 3u)
#define sf_float_flag_crash_overflow  (0x04 << 3u)
#define sf_float_flag_crash_divbyzero (0x08 << 3u)
#define sf_float_flag_crash_invalid   (0x10 << 3u)

/*----------------------------------------------------------------------------
| Software IEC/IEEE floating-point exception flags.
| bits 9 to 13
*----------------------------------------------------------------------------*/
#define sf_fpu_state_exception_mask (0x1f00)
#define sf_float_flag_inexact   (0x01 << 8u)
#define sf_float_flag_underflow (0x02 << 8u)
#define sf_float_flag_overflow  (0x04 << 8u)
#define sf_float_flag_divbyzero (0x08 << 8u)
#define sf_float_flag_invalid   (0x10 << 8u)

/*----------------------------------------------------------------------------
| Software FPU state.
*----------------------------------------------------------------------------*/
typedef sf_uint16 sf_fpu_state;
#define sf_fpu_state_default ((sf_fpu_state) (sf_float_tininess_after_rounding \
                                              | sf_float_round_nearest_even \
                                              | sf_float_flag_crash_divbyzero \
                                              | sf_float_flag_crash_invalid))

/*----------------------------------------------------------------------------
| Software IEC/IEEE floating-point function results carrying the FPU state.
*----------------------------------------------------------------------------*/
typedef struct {
    sf_flag result;
    sf_fpu_state fpu_state;
} sf_resultFlag;

typedef struct {
    sf_float32 result;
    sf_fpu_state fpu_state;
} sf_result32f;

typedef struct {
    sf_int32 result;
    sf_fpu_state fpu_state;
} sf_result32i;

typedef struct {
    sf_float64 result;
    sf_fpu_state fpu_state;
} sf_result64f;

typedef struct {
    sf_int64 result;
    sf_fpu_state fpu_state;
} sf_result64i;

/*----------------------------------------------------------------------------
| Routine to raise any or all of the software IEC/IEEE floating-point
| exception flags specified by `flags'.  Floating-point traps can be
| defined here if desired.  It is currently not possible for such a trap
| to substitute a result value.  If traps are not implemented, this routine
| should be simply `float_exception_flags |= flags;'.
*----------------------------------------------------------------------------*/
#define sf_float_raise(state,flags) do { (state) |= (flags); } while (0)

/*----------------------------------------------------------------------------
| Software IEC/IEEE integer-to-floating-point values of 1.0000:
*----------------------------------------------------------------------------*/
#define sf_float32_one 0x3f800000u
#define sf_float64_one SF_ULIT64(0x3ff0000000000000)

/*----------------------------------------------------------------------------
| Software IEC/IEEE integer-to-floating-point conversion routines.
*----------------------------------------------------------------------------*/
sf_result32f sf_int32_to_float32(sf_int32, sf_fpu_state);
sf_float64 sf_int32_to_float64(sf_int32);
sf_result32f sf_int64_to_float32(sf_int64, sf_fpu_state);
sf_result64f sf_int64_to_float64(sf_int64, sf_fpu_state);

/*----------------------------------------------------------------------------
| Software IEC/IEEE single-precision conversion routines.
*----------------------------------------------------------------------------*/
sf_result32i sf_float32_to_int32(sf_float32, sf_fpu_state);
sf_result32i sf_float32_to_int32_round_to_zero(sf_float32, sf_fpu_state);
sf_result64i sf_float32_to_int64(sf_float32, sf_fpu_state);
sf_result64i sf_float32_to_int64_round_to_zero(sf_float32, sf_fpu_state);
sf_result64f sf_float32_to_float64(sf_float32, sf_fpu_state);


/*----------------------------------------------------------------------------
| Software IEC/IEEE packing routines.
*----------------------------------------------------------------------------*/
sf_result32f sf_roundAndPackFloat32(sf_flag, sf_int16, sf_bits32, sf_fpu_state);
sf_result64f sf_roundAndPackFloat64(sf_flag, sf_int16, sf_bits64, sf_fpu_state);


/*----------------------------------------------------------------------------
| Software IEC/IEEE single-precision operations.
*----------------------------------------------------------------------------*/
sf_result32f sf_float32_round_to_int(sf_float32, sf_fpu_state);
inline sf_float32 sf_float32_neg(const sf_float32 n) {
    return n ^ (sf_bits32) 0x80000000;
}
sf_result32f sf_float32_add(sf_float32, sf_float32, sf_fpu_state);
sf_result32f sf_float32_sub(sf_float32, sf_float32, sf_fpu_state);
sf_result32f sf_float32_mul(sf_float32, sf_float32, sf_fpu_state);
sf_result32f sf_float32_div(sf_float32, sf_float32, sf_fpu_state);
sf_result32f sf_float32_rem(sf_float32, sf_float32, sf_fpu_state);
sf_result32f sf_float32_sqrt(sf_float32, sf_fpu_state);
sf_resultFlag sf_float32_eq(sf_float32, sf_float32, sf_fpu_state);
sf_resultFlag sf_float32_le(sf_float32, sf_float32, sf_fpu_state);
sf_resultFlag sf_float32_lt(sf_float32, sf_float32, sf_fpu_state);
sf_resultFlag sf_float32_eq_signaling(sf_float32, sf_float32, sf_fpu_state);
sf_resultFlag sf_float32_le_quiet(sf_float32, sf_float32, sf_fpu_state);
sf_resultFlag sf_float32_lt_quiet(sf_float32, sf_float32, sf_fpu_state);
sf_flag sf_float32_is_signaling_nan(sf_float32);

/*----------------------------------------------------------------------------
| Software IEC/IEEE double-precision conversion routines.
*----------------------------------------------------------------------------*/
sf_result32i sf_float64_to_int32(sf_float64, sf_fpu_state);
sf_result32i sf_float64_to_int32_round_to_zero(sf_float64, sf_fpu_state);
sf_result64i sf_float64_to_int64(sf_float64, sf_fpu_state);
sf_result64i sf_float64_to_int64_round_to_zero(sf_float64, sf_fpu_state);
sf_result32f sf_float64_to_float32(sf_float64, sf_fpu_state);

/*----------------------------------------------------------------------------
| Software IEC/IEEE double-precision operations.
*----------------------------------------------------------------------------*/
sf_result64f sf_float64_round_to_int(sf_float64, sf_fpu_state);
static inline sf_float64 sf_float64_neg(const sf_float64 n) {
    return n ^ (sf_bits64) SF_ULIT64(0x8000000000000000);
}
sf_result64f sf_float64_add(sf_float64, sf_float64, sf_fpu_state);
sf_result64f sf_float64_sub(sf_float64, sf_float64, sf_fpu_state);
sf_result64f sf_float64_mul(sf_float64, sf_float64, sf_fpu_state);
sf_result64f sf_float64_div(sf_float64, sf_float64, sf_fpu_state);
sf_result64f sf_float64_rem(sf_float64, sf_float64, sf_fpu_state);
sf_result64f sf_float64_sqrt(sf_float64, sf_fpu_state);
sf_resultFlag sf_float64_eq(sf_float64, sf_float64, sf_fpu_state);
sf_resultFlag sf_float64_le(sf_float64, sf_float64, sf_fpu_state);
sf_resultFlag sf_float64_lt(sf_float64, sf_float64, sf_fpu_state);
sf_resultFlag sf_float64_eq_signaling(sf_float64, sf_float64, sf_fpu_state);
sf_resultFlag sf_float64_le_quiet(sf_float64, sf_float64, sf_fpu_state);
sf_resultFlag sf_float64_lt_quiet(sf_float64, sf_float64, sf_fpu_state);
sf_flag sf_float64_is_signaling_nan(sf_float64);

#pragma GCC visibility pop

#ifdef __cplusplus
} /* extern "C" { */
#endif

#endif /* SHAREMIND_SOFTFLOAT_SOFTFLOAT_H */