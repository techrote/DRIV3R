/* MZ-018: row-major signed two's-complement INT4 GEMV; research-only CPU kernel.
 * No allocation, no FP32 weight expansion, no SIMD/BLAS dependency.
 * ABI: nibbles 0..7 => 0..7, nibbles 8..15 => -8..-1;
 * quantizer never emits -8, but kernel must decode it correctly.
 */
#include <stddef.h>
#include <stdint.h>

void mz018_i4_gemv(const uint8_t *packed, const float *scales,
                   const float *x, float *y, size_t rows, size_t cols) {
    if (!packed || !scales || !x || !y || !cols) return;
    size_t stride = (cols + 1u) / 2u;
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *row = packed + r * stride;
        float sum = 0.0f;
        size_t j = 0;
        for (; j + 1 < cols; j += 2) {
            const uint8_t b = row[j / 2];
            const int lo = (int)(b & 15u) - (((b & 15u) >= 8u) ? 16 : 0);
            const int hi = (int)(b >> 4) - ((b >> 4) >= 8u ? 16 : 0);
            sum += (float)lo * x[j] + (float)hi * x[j + 1];
        }
        if (j < cols) {
            int lo = (int)(row[j / 2] & 15u);
            if (lo >= 8) lo -= 16;
            sum += (float)lo * x[j];
        }
        y[r] = sum * scales[r];
    }
}

void mz018_f32_gemv(const float *w, const float *x, float *y,
                    size_t rows, size_t cols) {
    if (!w || !x || !y || !cols) return;
    for (size_t r = 0; r < rows; ++r) {
        const float *row = w + r * cols;
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) sum += row[c] * x[c];
        y[r] = sum;
    }
}
