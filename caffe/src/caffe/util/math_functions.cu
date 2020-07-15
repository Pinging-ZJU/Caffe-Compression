#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

// Compression idea kernel code
__global__ void pre_compression(int arraySize, int kernelSize, float arrayGPU[],
 int valueIndex[]) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int indexNumber;
    if(arraySize % kernelSize == 0)
        indexNumber = arraySize / kernelSize;
    else
        indexNumber = arraySize / kernelSize + 1;
    int index_begin = indexNumber*i;
    int index_end = index_begin + indexNumber;
    if(index_begin < arraySize){
        int valueFlag = 0;
        if(index_end > arraySize)
            index_end = arraySize;
        for(int cur= index_begin; cur<index_end; cur++){
            if(arrayGPU[cur] != 0)
                valueFlag++;
        }
        valueIndex[i] = valueFlag;
    }
}

// get compressed data size & return 
__global__ void compression_array_size(int valueIndex[], int kernelSize, int gpucompressedSize[],
 int compressedValueIndex[])
{
    gpucompressedSize[0] = 0;
    compressedValueIndex[0] = 0;
    for(int i=1; i<kernelSize; i++){
        gpucompressedSize[0] += valueIndex[i-1];
        compressedValueIndex[i] = gpucompressedSize[0];
    }
    gpucompressedSize[0] += valueIndex[kernelSize-1];
}

// compress data
__global__ void compression(int arraySize, int kernelSize, float arrayGPU[], float compressedList[],
 int compressedValueIndex[], uint32_t compressedBinIndex[])
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int indexNumber;
    if(arraySize % kernelSize == 0)
        indexNumber = arraySize / kernelSize;
    else
        indexNumber = arraySize / kernelSize + 1;
    int index_begin = indexNumber*i;
    int index_end = index_begin + indexNumber;
    int eachProcessIndexCount;
    if(indexNumber % 32 == 0)
        eachProcessIndexCount = indexNumber / 32;
    else
        eachProcessIndexCount = indexNumber / 32 + 1;
    if(index_begin < arraySize){
        int ValueIndex = compressedValueIndex[i];
        int cursor = 0;
        uint32_t  final, indexcut[32];
        uint32_t powerList[32] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 
            2048, 4096, 8192, 16384,32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 
                8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648 };
        if(index_end > arraySize)
            index_end = arraySize;
        int BinIndexCursor = i * eachProcessIndexCount;
        for(int cur= index_begin; cur<index_end; cur++){
            if(cursor==32){
                final = 0;
                cursor = 0;
                for (int j = 0; j < 32; j++) {
                    if (indexcut[j] == 1)
                        final += powerList[j];
                }
                compressedBinIndex[BinIndexCursor++] = final;
            }
            if (arrayGPU[cur] == 0){
                indexcut[cursor++] = 0;
            }
            else{
                indexcut[cursor++] = 1;
                compressedList[ValueIndex++] = arrayGPU[cur];
            }
        }
        final = 0;
        for (int j = 0; j < cursor; j++) {
            if (indexcut[j] == 1)
                final += powerList[j];
        }
        compressedBinIndex[BinIndexCursor] = final;
    }

}

// valueIndex & gpucompressedValueIndex need to be transfered into this function, to reduce the malloc times
// parameters: tensor gpu pointer / tensor cpu pointer / "pre_compression" process index needed / "compression_array_size" process index
// / gridsize / blocksize / tensor size / compressed tensor's size(gpu) / compressed tensor's size(cpu) / CPU 32 index
// void caffe_sparsity_compression(float* arrayGPU, float* arrayCPU, float *compressedList, int* valueIndex, int* gpucompressedValueIndex, int* cpucompressedValueIndex,
//  int gridsize, int blocksize, int tensor_size, int* gpucompressedSize, int* cpucompressedSize, uint32_t* CPUBinIndex, uint32_t* GPUBinIndex) {

void caffe_sparsity_compression(float* arrayGPU, float* arrayCPU, float **compressedList, int* valueIndex, int* gpucompressedValueIndex,
 int gridsize, int blocksize, int tensor_size, int* gpucompressedSize, int* cpucompressedSize, uint32_t** GPUBinIndex) {
  int process = gridsize * blocksize;

  dim3 dimGrid(gridsize);
  dim3 dimBlock(blocksize);

  pre_compression<<<dimGrid, dimBlock>>>(tensor_size, process, arrayGPU, valueIndex);
  compression_array_size<<<1, 1>>>(valueIndex, process, gpucompressedSize, gpucompressedValueIndex);
  cudaMemcpy((void*) cpucompressedSize, (void*) gpucompressedSize, sizeof(int) * 1, cudaMemcpyDeviceToHost);

  cudaMalloc((void**)compressedList, cpucompressedSize[0] * sizeof(float));

  int IndexCount = ceil(ceil(tensor_size / process) / 32) * process;
  // 必须在这里定义，IndexCount不能提前知道
  cudaMalloc((void**)GPUBinIndex, sizeof(int) * IndexCount);
  compression<<<dimGrid, dimBlock>>>(tensor_size, process, arrayGPU, *compressedList, gpucompressedValueIndex, *GPUBinIndex);
  // transfer 
  // cudaMemcpy((void*) arrayCPU, (void*) compressedList, sizeof(int) * cpucompressedSize[0], cudaMemcpyDeviceToHost);
  // GPU中的索引数据可以不用转出，因为数据量比较小
  // cudaMallocHost(&CPUBinIndex, sizeof(int) * IndexCount);
  // cudaMemcpy((void*) CPUBinIndex, (void*) GPUBinIndex, sizeof(int) * IndexCount, cudaMemcpyDeviceToHost);
  // cudaMemcpy((void*) cpucompressedValueIndex, (void*) gpucompressedValueIndex, sizeof(int) * process, cudaMemcpyDeviceToHost);
}

// decompression
__global__ void decompression(float arrayGPU[], float destiGPU[], int arraySize, int kernelSize,
 uint32_t gpuDataindex[], int beginIndex[])
{   
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int indexNumber, processOperateNum;
    if(arraySize % kernelSize == 0)
        processOperateNum = arraySize / kernelSize;
    else
        processOperateNum = arraySize / kernelSize + 1;
    if(processOperateNum * i < arraySize){
        if(processOperateNum % 32 ==0)
            indexNumber = processOperateNum / 32;
        else
            indexNumber = processOperateNum / 32 + 1;

        int index_begin = indexNumber * i;
        int index_end = index_begin + indexNumber;
        int WriteCur;
        int dataReadnow = beginIndex[i];
        uint32_t temp;
        int result;
        int cur;
        int binFlag = 0;
        for(cur= index_begin; cur<index_end-1; cur++){
            result = 0;
            temp = gpuDataindex[cur];
            WriteCur = (processOperateNum * i) + binFlag * 32;
            binFlag++;
            while(temp){
                result = temp % 2;
                temp = temp / 2;
                if(result==1){
                    destiGPU[WriteCur++] = arrayGPU[dataReadnow];
                    dataReadnow++;
                }
                else{
                    destiGPU[WriteCur++] = 0;
                }
            }
        }
        int threshold;
        int finalKernel;
        if(arraySize % processOperateNum == 0)
            finalKernel = arraySize / processOperateNum;
        else
            finalKernel = arraySize / processOperateNum + 1;
        if(i == finalKernel)
            threshold = arraySize;
        else
            threshold = (i+1) * processOperateNum;
        temp = gpuDataindex[cur];
        WriteCur = (processOperateNum * i) + binFlag * 32;
        while(temp){
            if(WriteCur==threshold)
                break;
            result = temp % 2;
            temp = temp / 2;
            if(result==1){
                destiGPU[WriteCur++] = arrayGPU[dataReadnow];
                dataReadnow++;
            }
            else{
                destiGPU[WriteCur++] = 0;
            }
        }
    }
}

// 数据先从arrayCPU转移到compressedList（转移cpucompressedSize[0]个），之后处理转到arrayGPU
void caffe_sparsity_decompression(float* arrayGPU, float* arrayCPU, float *compressedList, int* gpucompressedValueIndex,
  int gridsize, int blocksize, int tensor_size, int* cpucompressedSize, uint32_t* GPUBinIndex)
{

  dim3 dimGrid(gridsize);
  dim3 dimBlock(blocksize);
  int process = gridsize * blocksize;
  
  decompression<<<dimGrid, dimBlock>>>(compressedList, arrayGPU, tensor_size, process, GPUBinIndex, gpucompressedValueIndex);
  cudaFree(compressedList);
} 

}  // namespace caffe
