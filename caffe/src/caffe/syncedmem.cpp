#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include <unistd.h>
#include<sys/time.h>


namespace caffe {
SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  check_device();
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::Sleep(int ms)
{
    struct timeval delay;
    delay.tv_sec = 0;
    delay.tv_usec = ms * 1000; // 20 ms
    select(0, NULL, NULL, NULL, &delay);
    return (const void*)gpu_ptr_;
}

double SyncedMemory::get_cur_time_ms() {
    struct timeval   tv;
    struct timezone  tz;
    double cur_time;
    gettimeofday(&tv, &tz);
    cur_time = tv.tv_sec * 1000 + tv.tv_usec / 1000.0;
    return cur_time;
}


const void* SyncedMemory::gpufree() {
    cudaFree(gpu_ptr_);
}

// Cping-DIY 将GPU中数据异步转移至CPU
const void* SyncedMemory::async_gpu2cpu(int size_) {
    check_device();
    cudaStream_t STREAM_GPU_to_CPU_;
    cudaStreamCreate(&STREAM_GPU_to_CPU_);
    const cudaMemcpyKind put = cudaMemcpyDeviceToHost;
#ifndef CPU_ONLY

    if (gpu_ptr_ != NULL && cpu_ptr_ == NULL) {
        CaffeMallocHost(&cpu_ptr_, size_ * 4, &cpu_malloc_use_cuda_);
        own_cpu_data_ = true;
    }
    int deley = ceil(float(size_ * 4) / 1024 / 1024 / 24);
    
    
    //CUDA_CHECK(cudaMemcpyAsync(cpu_ptr_, gpu_ptr_, size_ * 4, put, STREAM_GPU_to_CPU_));
    //double t1 = get_cur_time_ms();
    Sleep(deley);
    CUDA_CHECK(cudaMemcpyAsync(cpu_ptr_, gpu_ptr_, size_ * 4, put, STREAM_GPU_to_CPU_));
    //double t2 = get_cur_time_ms();
    
    //printf("Size %f mb -- delay time is %lf\n",  float(size_)*4 /1024/1024, t2 - t1);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    
    return (const void*)gpu_ptr_;
}

// Cping-DIY 将CPU中数据异步转移至GPU
const void* SyncedMemory::async_cpu2gpu(int size_) {
    check_device();
    cudaStream_t STREAM_CPU_to_GPU_;
    cudaStreamCreate(&STREAM_CPU_to_GPU_);
    const cudaMemcpyKind put = cudaMemcpyHostToDevice;
    int deley = ceil(float(size_ * 4) / 1024 / 1024 / 24);
    Sleep(deley);
    CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_ * 4, put, STREAM_CPU_to_GPU_));
    //printf("Size %f mb \n", float(size_) * 4 / 1024 / 1024);
    return (const void*)cpu_ptr_;
}

/*
    compression and decompression function location
*/
const void* SyncedMemory::compression(int size_) {
    //check_device();
    int gridsize = 64;
    int blocksize = 128;
#ifndef CPU_ONLY

    if (gpu_ptr_ != NULL && cpu_ptr_ == NULL) {
        CaffeMallocHost(&cpu_ptr_, size_ * sizeof(float), &cpu_malloc_use_cuda_);
        own_cpu_data_ = true;
    }
    head_ = SYNCED;
#else
    NO_GPU;
#endif

    if (!compression_flag) {
        // 可定制，暂时先这么设计
        int process = gridsize * blocksize;
        // 每个参数只分配一次空间，后面沿用即可
        cudaMalloc((void**)&valueIndex, sizeof(int) * process);
        cudaMalloc((void**)&gpucompressedValueIndex, sizeof(int) * process);
        cudaMalloc((void**)&gpucompressedSize, sizeof(int) * 1);
        cudaMallocHost(&cpucompressedSize, sizeof(int) * 1);
        compression_flag = true;
    }
    caffe::caffe_sparsity_compression((float*)gpu_ptr_, (float*)cpu_ptr_, &compressedList, valueIndex, gpucompressedValueIndex,
        gridsize, blocksize, size_, gpucompressedSize, cpucompressedSize, &GPUBinIndex);

    // 将压缩后的数据转移至CPU中，并释放相关中间值
    cudaStream_t STREAM_GPU_to_CPU_;
    cudaStreamCreate(&STREAM_GPU_to_CPU_);
    //printf("------------------------------Data size is : %d  %d   ratio is %f-----------------------------------\n", cpucompressedSize[0], size_, float(cpucompressedSize[0])/float(size_) *100);
    int deley = ceil(float(cpucompressedSize[0] * 4) / 1024 / 1024 / 24);
    Sleep(deley);
    CUDA_CHECK(cudaMemcpyAsync((float*)cpu_ptr_, (float*)compressedList, sizeof(float) * cpucompressedSize[0], cudaMemcpyDeviceToHost, STREAM_GPU_to_CPU_));
    //cudaFree(compressedList);
    return (const void*)gpu_ptr_;
}
/*
    decompression function
*/
const void* SyncedMemory::decompression(int size_) {
    //check_device();
    int gridsize = 64;
    int blocksize = 128;
    //cudaMalloc((void**)&compressedList, cpucompressedSize[0] * sizeof(float));
    caffe::caffe_sparsity_decompression((float*)gpu_ptr_, (float*)cpu_ptr_, compressedList, gpucompressedValueIndex,
        gridsize, blocksize, size_, cpucompressedSize, GPUBinIndex);
    return (const void*)gpu_ptr_;
}

const void* SyncedMemory::decompression_cpu2gpu_asyc_transfer() {
    cudaStream_t STREAM_CPU_to_GPU_;
    cudaStreamCreate(&STREAM_CPU_to_GPU_);
    // 将CPU数据转移到GPU中
    int deley = ceil(float(cpucompressedSize[0] * 4) / 1024 / 1024 / 24);
    Sleep(deley);
    CUDA_CHECK(cudaMemcpyAsync((float*)compressedList, (float*)cpu_ptr_, sizeof(float) * cpucompressedSize[0], cudaMemcpyHostToDevice, STREAM_CPU_to_GPU_));
    return (const void*)gpu_ptr_;
}



const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

}  // namespace caffe

