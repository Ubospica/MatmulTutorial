#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
__forceinline__ __device__ unsigned int
cast_smem_ptr_to_int(const void* const smem_ptr)
{
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long


__global__ void matmul(half *A, half *B, half *C, int M, int N, int K, float alpha, float beta) {
    half* compute = C;
//
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> matmul_reindex_shared_dyn_wmma_accumulator[16];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_reindex_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> T_transpose_reindex_shared_dyn_wmma_matrix_b[4];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      if (((((((int)blockIdx.x) % 352) >> 4) * 2) + ((((int)blockIdx.x) & 3) >> 1)) < 43) {
        nvcuda::wmma::fill_fragment(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_3_init * 4) + ax2_0_3_init)], 0.000000e+00f);
      }
    }
  }
  // #pragma unroll
  for (int ax3_0_0 = 0; ax3_0_0 < 128; ++ax3_0_0) {
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {
      *(uint4*)(((half*)buf_dyn_shmem) + (((((((ax0_ax1_fused_0 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8)) + 6144)) = *(uint4*)(A + (((((((((((int)blockIdx.x) / 352) * 2097152) + (((((int)blockIdx.x) & 15) >> 2) * 524288)) + (ax0_ax1_fused_0 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax3_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1) {
      if (((((((int)blockIdx.x) % 352) >> 4) * 2) + ((((int)blockIdx.x) & 3) >> 1)) < 43) {
        *(uint4*)(((half*)buf_dyn_shmem) + ((((((ax0_ax1_fused_0_1 * 1536) + (((int)threadIdx.z) * 768)) + (((((int)threadIdx.x) & 3) >> 1) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B + ((((((((((((int)blockIdx.x) % 352) >> 4) * 2097152) + ((((int)blockIdx.x) & 3) * 524288)) + (ax0_ax1_fused_0_1 * 131072)) + (((int)threadIdx.z) * 65536)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax3_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
      }
    }
    __syncthreads();
    for (int ax3_0_1 = 0; ax3_0_1 < 2; ++ax3_0_1) {
      for (int ax0 = 0; ax0 < 4; ++ax0) {
        nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[ax0], (&(((half*)buf_dyn_shmem)[((((((int)threadIdx.z) * 3072) + (ax0 * 768)) + (ax3_0_1 * 384)) + 6144)])), (int64_t)24);
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
        nvcuda::wmma::load_matrix_sync(T_transpose_reindex_shared_dyn_wmma_matrix_b[ax0_1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 3072) + (ax0_1 * 768)) + (ax3_0_1 * 384))])), (int64_t)24);
      }
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {
          if (((((((int)blockIdx.x) % 352) >> 4) * 2) + ((((int)blockIdx.x) & 3) >> 1)) < 43) {
            nvcuda::wmma::mma_sync(matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)], A_reindex_shared_dyn_wmma_matrix_a[ax1_0_3], T_transpose_reindex_shared_dyn_wmma_matrix_b[ax2_0_3], matmul_reindex_shared_dyn_wmma_accumulator[((ax1_0_3 * 4) + ax2_0_3)]);
          }
        }
      }
    }
  }
  __syncthreads();
  for (int ax1 = 0; ax1 < (min(((688 - ((((int)blockIdx.x) & 3) * 8)) - (((((int)blockIdx.x) % 352) >> 4) * 32)), 8) >> 3); ++ax1) {
    for (int ax2_1 = 0; ax2_1 < 4; ++ax2_1) {
      for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
        nvcuda::wmma::store_matrix_sync((&(((float*)buf_dyn_shmem)[(((((ax1 * 16384) + (((int)threadIdx.z) * 8192)) + (ax2_1 * 2048)) + (((int)threadIdx.y) * 1024)) + (ax3_1 * 256))])), matmul_reindex_shared_dyn_wmma_accumulator[(((ax1 * 16) + (ax2_1 * 4)) + ax3_1)], (int64_t)16, nvcuda::wmma::mem_row_major);
      }
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_2 = 0; ax0_ax1_fused_0_2 < 16; ++ax0_ax1_fused_0_2) {
    if (((((((int)blockIdx.x) % 352) >> 4) * 2) + ((((int)blockIdx.x) & 3) >> 1)) < 43) {
      uint4 __1;
      ulonglong4 v_ = *(ulonglong4*)(((float*)buf_dyn_shmem) + (((((((((ax0_ax1_fused_0_2 * 8) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) >> 4) * 2048) + (((((int)threadIdx.x) & 15) >> 1) * 256)) + ((((((ax0_ax1_fused_0_2 * 8) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) & 15) * 16)) + ((((int)threadIdx.x) & 1) * 8)));
      ((half2*)(&(__1.x)))->x = (half)(((float2*)(&(v_.x)))->x);
      ((half2*)(&(__1.x)))->y = (half)(((float2*)(&(v_.x)))->y);
      ((half2*)(&(__1.y)))->x = (half)(((float2*)(&(v_.y)))->x);
      ((half2*)(&(__1.y)))->y = (half)(((float2*)(&(v_.y)))->y);
      ((half2*)(&(__1.z)))->x = (half)(((float2*)(&(v_.z)))->x);
      ((half2*)(&(__1.z)))->y = (half)(((float2*)(&(v_.z)))->y);
      ((half2*)(&(__1.w)))->x = (half)(((float2*)(&(v_.w)))->x);
      ((half2*)(&(__1.w)))->y = (half)(((float2*)(&(v_.w)))->y);
      *(uint4*)(compute + ((((((((((((int)blockIdx.x) / 352) * 5636096) + (((((int)blockIdx.x) & 15) >> 2) * 1409024)) + (ax0_ax1_fused_0_2 * 88064)) + (((int)threadIdx.z) * 44032)) + (((int)threadIdx.y) * 22016)) + ((((int)threadIdx.x) >> 4) * 11008)) + (((((int)blockIdx.x) % 352) >> 4) * 512)) + ((((int)blockIdx.x) & 3) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = __1;
    }
  }
}
