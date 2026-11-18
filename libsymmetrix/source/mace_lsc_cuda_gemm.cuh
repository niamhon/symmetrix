#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace lsc_cuda {
// Polar's high/low FP16 tensor-core GEMM, unchanged apart from namespace/macro.
// FP32 fallbacks and the dispatch heuristic live in DeviceData::gemm.
struct GemmDesc {
    const float* A;
    const __half *Bh, *Bl;
    float* C;
    int M, N, K, lda, ldc, tile0, shift;
    float alpha, beta;
};
struct GemmBatch { GemmDesc g[32]; int n; };

__host__ __device__ inline int scale_shift(float a) {   // a 2^shift in [2^12, 2^13)
    if (!(a > 0)) return 0;
    int e;
    frexpf(a, &e);
    return 13 - e;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define LSC_TENSOR_CORES
#endif

#ifdef LSC_TENSOR_CORES
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
                 "l"(gmem), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void split(float2 v, float scale, uint32_t& hi, uint32_t& lo) {
    const __half hx = __float2half_rn(v.x*scale), hy = __float2half_rn(v.y*scale);
    const __half2 h = __halves2half2(hx, hy);
    const __half2 l = __halves2half2(__float2half_rn(v.x*scale - __half2float(hx)),
                                     __float2half_rn(v.y*scale - __half2float(hy)));
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
__device__ __forceinline__ void row_scale(float m, float& scale, float& unscale) {
    m = fmaxf(m, __shfl_xor_sync(~0u, m, 1));
    m = fmaxf(m, __shfl_xor_sync(~0u, m, 2));
    const int s = min(max(12 - int((__float_as_uint(m) >> 23) & 255) + 127, -100), 100);
    scale = __uint_as_float(uint32_t(127 + s) << 23);
    unscale = __uint_as_float(uint32_t(127 - s) << 23);
}
__device__ __forceinline__ float amax4(float2 a, float2 b) { return fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)),
        fmaxf(fabsf(b.x), fabsf(b.y))); }
__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                 "r"(b[0]), "r"(b[1]));
}
#endif

constexpr int SA = 24;
constexpr int SB = 24;
template <int BM, int BN> constexpr int gemm_smem = 3*(sizeof(float)*BM*SA + 2*sizeof(__half)*BN*SB);

__device__ __forceinline__ void store(const GemmDesc& d, int r, int c, float x, float y, float u) {
    float2* p = reinterpret_cast<float2*>(d.C + (size_t)r*d.ldc + c);
    float2 v = make_float2(x*u, y*u);
    if (d.beta != 0) {
        const float2 o = *p;
        v.x += d.beta*o.x;
        v.y += d.beta*o.y;
    }
    *p = v;
}

template <int BM, int BN, int WM, int WN> __global__ void __launch_bounds__(32*WM*WN, 2) k_gemm(GemmBatch batch) {
#ifdef LSC_TENSOR_CORES
    constexpr int NT = 32*WM*WN, TM = BM/WM/16, TN = BN/WN/8;
    extern __shared__ __align__(16) unsigned char smem[];
    auto As = reinterpret_cast<float (*)[BM][SA]>(smem);
    auto Bh = reinterpret_cast<__half (*)[BN][SB]>(smem + sizeof(float)*3*BM*SA);
    auto Bl = reinterpret_cast<__half (*)[BN][SB]>(smem + sizeof(float)*3*BM*SA + sizeof(__half)*3*BN*SB);
    int gi = 0;
    while (gi + 1 < batch.n && batch.g[gi + 1].tile0 <= int(blockIdx.x)) ++gi;
    const GemmDesc d = batch.g[gi];
    const int tile = blockIdx.x - d.tile0;
    const int m0 = tile/(d.N/BN)*BM;
    const int n0 = tile%(d.N/BN)*BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = warp/WN;
    const int wn = warp%WN;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const float unscale = d.alpha*ldexpf(1.f, -d.shift);
    float acc[TM][TN][4] = {};
    const int ktiles = d.K/16;
    auto load = [&](int st, int kt) {
        for (int c = threadIdx.x; c < BM*4; c += NT) {
            const int r = c >> 2;
            const int gr = m0 + r;
            cp_async16(&As[st][r][(c & 3)*4], d.A + (size_t)(gr < d.M ? gr : 0)*d.lda + kt*16 + (c & 3)*4, gr < d.M);
        }
        for (int c = threadIdx.x; c < BN*2; c += NT) {
            const size_t g = (size_t)(n0 + (c >> 1))*d.K + kt*16 + (c & 1)*8;
            cp_async16(&Bh[st][c >> 1][(c & 1)*8], d.Bh + g, true);
            cp_async16(&Bl[st][c >> 1][(c & 1)*8], d.Bl + g, true);
        }
        asm volatile("cp.async.commit_group;");
    };
    load(0, 0);
    if (ktiles > 1)
        load(1, 1);
    else
        asm volatile("cp.async.commit_group;");
    for (int kt = 0; kt < ktiles; ++kt) {
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        if (kt + 2 < ktiles)
            load((kt + 2)%3, kt + 2);
        else
            asm volatile("cp.async.commit_group;");
        const int st = kt%3;
        uint32_t ah[TM][4], al[TM][4], bh[TN][2], bl[TN][2];
        float us[TM][2];
        for (int i = 0; i < TM; ++i) {
            const int r = wm*(BM/WM) + i*16 + gid;
            const float2 a0 = *reinterpret_cast<const float2*>(&As[st][r][2*tig]);
            const float2 a1 = *reinterpret_cast<const float2*>(&As[st][r + 8][2*tig]);
            const float2 a2 = *reinterpret_cast<const float2*>(&As[st][r][2*tig + 8]);
            const float2 a3 = *reinterpret_cast<const float2*>(&As[st][r + 8][2*tig + 8]);
            float s0, s1;
            row_scale(amax4(a0, a2), s0, us[i][0]);
            row_scale(amax4(a1, a3), s1, us[i][1]);
            split(a0, s0, ah[i][0], al[i][0]);
            split(a1, s1, ah[i][1], al[i][1]);
            split(a2, s0, ah[i][2], al[i][2]);
            split(a3, s1, ah[i][3], al[i][3]);
        }
        for (int j = 0; j < TN; ++j) {
            const int n = wn*(BN/WN) + j*8 + gid;
            bh[j][0] = *reinterpret_cast<const uint32_t*>(&Bh[st][n][2*tig]);
            bh[j][1] = *reinterpret_cast<const uint32_t*>(&Bh[st][n][2*tig + 8]);
            bl[j][0] = *reinterpret_cast<const uint32_t*>(&Bl[st][n][2*tig]);
            bl[j][1] = *reinterpret_cast<const uint32_t*>(&Bl[st][n][2*tig + 8]);
        }
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j) {
                float part[4] = {};
                mma(part, al[i], bh[j]);
                mma(part, ah[i], bl[j]);
                mma(part, ah[i], bh[j]);
                for (int q = 0; q < 4; ++q) acc[i][j][q] += part[q]*us[i][q/2];
            }
    }
    asm volatile("cp.async.wait_group 0;");
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            const int r = m0 + wm*(BM/WM) + i*16 + gid, c = n0 + wn*(BN/WN) + j*8 + 2*tig;
            if (r < d.M) store(d, r, c, acc[i][j][0], acc[i][j][1], unscale);
            if (r + 8 < d.M) store(d, r + 8, c, acc[i][j][2], acc[i][j][3], unscale);
        }
#endif
}

template <int BM, int BN, int WM, int WN> void gemm_batch(GemmBatch b, cudaStream_t s)
{
    int tiles = 0;
    for (int i = 0; i < b.n; ++i) {
        b.g[i].tile0 = tiles;
        tiles += (b.g[i].M + BM - 1)/BM*(b.g[i].N/BN);
    }
    if (not tiles)
        return;
    k_gemm<BM, BN, WM, WN><<<tiles, 32*WM*WN, gemm_smem<BM, BN>, s>>>(b);
}
void gemm_batches(std::vector<GemmDesc>& pending, int num_sms, cudaStream_t s)
{
    for (size_t i0 = 0; i0 < pending.size(); i0 += 32) {
        GemmBatch b{};
        bool n128 = true;
        bool n64 = true;
        long big = 0;
        for (size_t i = i0; i < std::min(pending.size(), i0 + 32); ++i) {
            b.g[b.n++] = pending[i];
            n128 = n128 and pending[i].N%128 == 0;
            n64 = n64 and pending[i].N%64 == 0;
            big += (pending[i].M + 127)/128*(pending[i].N/32);
        }
        if (big < 4*num_sms)
            gemm_batch<32, 32, 1, 1>(b, s);
        else if (n128)
            gemm_batch<128, 128, 4, 2>(b, s);
        else if (n64)
            gemm_batch<128, 64, 4, 2>(b, s);
        else
            gemm_batch<128, 32, 4, 1>(b, s);
    }
    pending.clear();
}


#undef LSC_TENSOR_CORES
} // namespace lsc_cuda
