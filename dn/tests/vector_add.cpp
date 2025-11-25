#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>
#include <cmath>
#include <chrono>

// Helper function to get current time point
std::chrono::high_resolution_clock::time_point getTimeNow()
{
    return std::chrono::high_resolution_clock::now();
}

// Helper function to calculate elapsed time in milliseconds
long getElapsedTimeMs(std::chrono::high_resolution_clock::time_point start)
{
    auto end = getTimeNow();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return duration.count();
}

// Structure to hold parameters for each GPU thread
struct GPUThreadParams {
    int gpuId;
    hipStream_t *stream;
    float **d_A;
    float **d_B;
    float **d_C;
    const float *h_A;
    const float *h_B;
    float *h_C;
    size_t size;
    int dataOffset;
    int numElements;
    int threadsPerBlock;
};

// HIP kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

// Helper function to get device count with error checking
int getDeviceCount()
{
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to get device count (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return deviceCount;
}

// Helper function to print device properties with error checking
void printDeviceProperties(int deviceId)
{
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, deviceId);
    if(err != hipSuccess)
    {
        fprintf(stderr, "Failed to get device properties for GPU %d (error code %s)!\n", deviceId, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("GPU %d: %s\n", deviceId, prop.name);
}

// Helper function to allocate pinned host memory with error checking
void allocateHostMemory(float **h_ptr, size_t size, const char *vectorName)
{
    hipError_t err = hipHostMalloc((void**)h_ptr, size, hipHostMallocDefault);
    if (err != hipSuccess || *h_ptr == NULL)
    {
        fprintf(stderr, "Failed to allocate pinned host memory for vector %s (error code %s)!\n", vectorName, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to initialize vectors with random values
void initializeVectors(float *A, float *B, int numElements)
{
    for (int i = 0; i < numElements; ++i)
    {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }
}

// Helper function to setup host-side memory and initialize vectors
void setupHost(float **h_A, float **h_B, float **h_C, size_t size, int numElements)
{
    auto setup_start = getTimeNow();
    
    // Allocate host memory
    allocateHostMemory(h_A, size, "A");
    allocateHostMemory(h_B, size, "B");
    allocateHostMemory(h_C, size, "C");
    
    // Initialize input vectors
    printf("\nInitializing input vectors...\n");
    initializeVectors(*h_A, *h_B, numElements);
    
    long setup_time_ms = getElapsedTimeMs(setup_start);
    printf("Host setup time: %ld ms\n", setup_time_ms);
}

// Helper function to set device with error checking
void setDevice(int deviceId)
{
    hipError_t err = hipSetDevice(deviceId);
    if(err != hipSuccess)
    {
        fprintf(stderr, "Failed to set device %d (error code %s)!\n", deviceId, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to create stream with error checking
void createStream(hipStream_t *stream, int streamId)
{
    hipError_t err = hipStreamCreate(stream);
    if(err != hipSuccess)
    {
        fprintf(stderr, "Failed to create stream %d (error code %s)!\n", streamId, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to allocate device memory with error checking
void allocateDeviceMemory(float **d_ptr, size_t size, const char *vectorName, int gpuId)
{
    hipError_t err = hipMalloc((void **)d_ptr, size);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector %s on GPU %d (error code %s)!\n", 
                vectorName, gpuId, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to copy memory asynchronously with error checking
void copyMemoryAsync(void *dst, const void *src, size_t size, hipMemcpyKind kind, hipStream_t stream, const char *description, int gpuId)
{
    hipError_t err = hipMemcpyAsync(dst, src, size, kind, stream);
    if(err != hipSuccess)
    {
        fprintf(stderr, "Failed to %s GPU %d (error code %s)!\n", 
                description, gpuId, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to launch kernel with error checking
void launchKernel(float *d_A, float *d_B, float *d_C, int numElements, int threadsPerBlock, hipStream_t stream, int gpuId)
{
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("GPU %d: Launching %d blocks of %d threads\n", gpuId, blocksPerGrid, threadsPerBlock);
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, d_A, d_B, d_C, numElements);
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to launch kernel on GPU %d (error code %s)!\n", gpuId, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to synchronize stream with error checking
void synchronizeStream(hipStream_t stream, int streamId)
{
    printf("Waiting for GPU %d to complete...\n", streamId);
    hipError_t err = hipStreamSynchronize(stream);
    if(err != hipSuccess)
    {
        fprintf(stderr, "Failed to synchronize stream %d (error code %s)!\n", streamId, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("GPU %d done!\n", streamId);
}

// Helper function to free device memory with error checking
void freeDeviceMemory(void *d_ptr, const char *vectorName, int gpuId)
{
    hipError_t err = hipFree(d_ptr);
    if(err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector %s on GPU %d (error code %s)!\n", 
                vectorName, gpuId, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to destroy stream with error checking
void destroyStream(hipStream_t stream, int streamId)
{
    hipError_t err = hipStreamDestroy(stream);
    if(err != hipSuccess)
    {
        fprintf(stderr, "Failed to destroy stream %d (error code %s)!\n", streamId, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to free pinned host memory with error checking
void freeHostMemory(void *h_ptr, const char *vectorName)
{
    hipError_t err = hipHostFree(h_ptr);
    if(err != hipSuccess)
    {
        fprintf(stderr, "Failed to free pinned host memory for vector %s (error code %s)!\n", vectorName, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to setup GPU with stream creation, memory allocation, and data transfer
void setupGPU(int gpuId, hipStream_t *stream, float **d_A, float **d_B, float **d_C, const float *h_A, const float *h_B, size_t size, int dataOffset)
{
    printf("\n=== Setting up GPU %d ===\n", gpuId);
    createStream(stream, gpuId);
    // Allocate device memory
    allocateDeviceMemory(d_A, size, "A", gpuId);
    allocateDeviceMemory(d_B, size, "B", gpuId);
    allocateDeviceMemory(d_C, size, "C", gpuId);
    
    printf("Allocated %.2f MB on GPU %d\n", (size * 3) / (1024.0f * 1024.0f), gpuId);
    
    // Copy data to GPU (asynchronously)
    copyMemoryAsync(*d_A, h_A + dataOffset, size, hipMemcpyHostToDevice, *stream, "copy data to", gpuId);
    copyMemoryAsync(*d_B, h_B + dataOffset, size, hipMemcpyHostToDevice, *stream, "copy data to", gpuId);
}

// Helper function to retrieve results from GPU
void retrieveResults(int gpuId, hipStream_t stream, float *h_C, float *d_C, size_t size, int dataOffset)
{
    printf("\n=== Copying results back from GPU %d to host ===\n", gpuId);
    copyMemoryAsync(h_C + dataOffset, d_C, size, hipMemcpyDeviceToHost, stream, "copy results from", gpuId);
}

// Helper function to verify results and measure verification time
bool verifyResults(const float *h_A, const float *h_B, const float *h_C, int numElements)
{
    auto verify_start = getTimeNow();
    printf("\n=== Verifying results ===\n");
    
    bool passed = true;
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d! Expected %.5f, got %.5f\n", 
                    i, h_A[i] + h_B[i], h_C[i]);
            passed = false;
            break;
        }
    }
    
    long verify_time_ms = getElapsedTimeMs(verify_start);
    printf("Verification time: %ld ms\n", verify_time_ms);
    
    return passed;
}

// Thread function to handle all operations for one GPU
void gpuThreadFunc(GPUThreadParams *p)
{
    // printf start of GPU thread function
    printf("Starting GPU thread function for GPU %d\n", p->gpuId);
    
    setDevice(p->gpuId);
    
    setupGPU(p->gpuId, p->stream, p->d_A, p->d_B, p->d_C, p->h_A, p->h_B, p->size, p->dataOffset);
    
    auto kernel_start = getTimeNow();
    
    launchKernel(*p->d_A, *p->d_B, *p->d_C, p->numElements, p->threadsPerBlock, *p->stream, p->gpuId);
    synchronizeStream(*p->stream, p->gpuId);
    
    long kernel_time_ms = getElapsedTimeMs(kernel_start);
    printf("GPU %d kernel execution time: %ld ms\n", p->gpuId, kernel_time_ms);
    
    retrieveResults(p->gpuId, *p->stream, p->h_C, *p->d_C, p->size, p->dataOffset);
}

int main(int argc, char *argv[])
{
    // Print header
    printf("[Vector Addition using Multiple AMD GPUs with HIP]\n");

    // Parse command-line arguments for max GPU count
    int maxGPUs = -1; // -1 means use all available GPUs
    if (argc > 1)
    {
        maxGPUs = atoi(argv[1]);
        if (maxGPUs <= 0)
        {
            fprintf(stderr, "Invalid max GPU count: %s. Must be a positive integer.\n", argv[1]);
            fprintf(stderr, "Usage: %s [max_gpus]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    
    // Check number of available GPUs
    int availableGPUs = getDeviceCount();
    printf("Found %d GPU(s) available\n", availableGPUs);
    
    if (availableGPUs < 1)
    {
        fprintf(stderr, "This program requires at least 1 GPU, but none found!\n");
        exit(EXIT_FAILURE);
    }
    
    // Determine actual number of GPUs to use
    int deviceCount;
    if (maxGPUs > 0)
    {
        deviceCount = (maxGPUs < availableGPUs) ? maxGPUs : availableGPUs;
        printf("Using %d GPU(s) as requested (max: %d)\n", deviceCount, maxGPUs);
    }
    else
    {
        deviceCount = availableGPUs;
        printf("Using all %d GPU(s)\n", deviceCount);
    }

    // Print device properties
    for (int i = 0; i < deviceCount; i++)
    {
        printDeviceProperties(i);
    }

    // Number of elements in arrays
    int numElements = 100000000;
    size_t size = numElements * sizeof(float);
    printf("\nTotal vector size: %d elements (%.2f MB per array)\n", numElements, size / (1024.0f * 1024.0f));

    // Calculate split: divide work between N GPUs
    int elementsPerGPU = numElements / deviceCount;
    int remainder = numElements % deviceCount;
    
    // Calculate elements and sizes for each GPU
    std::vector<int> numElementsPerGPU(deviceCount);
    std::vector<size_t> sizePerGPU(deviceCount);
    std::vector<int> dataOffsets(deviceCount);
    
    int currentOffset = 0;
    for (int i = 0; i < deviceCount; i++)
    {
        // Distribute remainder across first few GPUs
        numElementsPerGPU[i] = elementsPerGPU + (i < remainder ? 1 : 0);
        sizePerGPU[i] = numElementsPerGPU[i] * sizeof(float);
        dataOffsets[i] = currentOffset;
        currentOffset += numElementsPerGPU[i];
        
        printf("GPU %d will process: %d elements (%.2f MB)\n", i, numElementsPerGPU[i], sizePerGPU[i] / (1024.0f * 1024.0f));
    }

    // Dynamic allocation for streams and device pointers
    std::vector<hipStream_t> streams(deviceCount);
    float *h_A = NULL, *h_B = NULL, *h_C = NULL; // Host memory
    std::vector<float*> d_A(deviceCount, nullptr); // Device pointers for input A
    std::vector<float*> d_B(deviceCount, nullptr); // Device pointers for input B
    std::vector<float*> d_C(deviceCount, nullptr); // Device pointers for output C

    // Setup host memory (main thread)
    setupHost(&h_A, &h_B, &h_C, size, numElements);

    // ============== Setup Parameters for GPU Threads ==============
    int threadsPerBlock = 256;
    
    std::vector<GPUThreadParams> params(deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        params[i] = {
            .gpuId = i,
            .stream = &streams[i],
            .d_A = &d_A[i],
            .d_B = &d_B[i],
            .d_C = &d_C[i],
            .h_A = h_A,
            .h_B = h_B,
            .h_C = h_C,
            .size = sizePerGPU[i],
            .dataOffset = dataOffsets[i],
            .numElements = numElementsPerGPU[i],
            .threadsPerBlock = threadsPerBlock
        };
    }

    // ============== Launch GPU Threads (Parallel Execution) ==============
    printf("\n=== Launching GPU threads for parallel execution ===\n");
    
    // Measure total parallel execution time
    auto parallel_start = getTimeNow();
    
    std::vector<std::thread> gpuThreads;
    for (int i = 0; i < deviceCount; i++)
    {
        gpuThreads.emplace_back(gpuThreadFunc, &params[i]);
    }
    
    // Wait for all GPU threads to complete
    printf("Main thread waiting for GPU threads to complete...\n");
    for (int i = 0; i < deviceCount; i++)
    {
        gpuThreads[i].join();
    }
    
    long parallel_time_ms = getElapsedTimeMs(parallel_start);
    
    printf("All GPU threads completed!\n");
    printf("Total parallel execution time (%d GPUs): %ld ms\n", deviceCount, parallel_time_ms);

    // ============== Verify Results ==============
    bool passed = verifyResults(h_A, h_B, h_C, numElements);

    if (passed)
    {
        printf("Test PASSED - All %d elements verified correctly!\n", numElements);
    }
    else
    {
        printf("Test FAILED\n");
    }

    // ============== Cleanup ==============
    printf("\n=== Cleaning up ===\n");
        
    // Free device memory for all GPUs
    for (int i = 0; i < deviceCount; i++)
    {
        setDevice(i);
        freeDeviceMemory(d_A[i], "A", i);
        freeDeviceMemory(d_B[i], "B", i);
        freeDeviceMemory(d_C[i], "C", i);
    }

    // Destroy all streams
    for (int i = 0; i < deviceCount; i++)
    {
        destroyStream(streams[i], i);
    }

    // Free pinned host memory
    freeHostMemory(h_A, "A");
    freeHostMemory(h_B, "B");
    freeHostMemory(h_C, "C");

    printf("Done\n");
    return passed ? 0 : 1;
}

