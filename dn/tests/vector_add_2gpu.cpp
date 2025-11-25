#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

// Helper function to allocate host memory with error checking
void allocateHostMemory(float **h_ptr, size_t size, const char *vectorName)
{
    *h_ptr = (float *)malloc(size);
    if (*h_ptr == NULL)
    {
        fprintf(stderr, "Failed to allocate host vector %s!\n", vectorName);
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

int main(void)
{
    // Print header
    printf("[Vector Addition using 2 AMD GPUs with HIP]\n");

    
    // Check number of available GPUs
    int deviceCount = getDeviceCount();

    printf("Found %d GPU(s)\n", deviceCount);
    
    if (deviceCount < 2)
    {
        fprintf(stderr, "This program requires at least 2 GPUs, but only %d found!\n", deviceCount);
        exit(EXIT_FAILURE);
    }

    // Print device properties
    for (int i = 0; i < 2; i++)
    {
        printDeviceProperties(i);
    }

    // Number of elements in arrays
    int numElements = 100000000;
    size_t size = numElements * sizeof(float);
    printf("\nTotal vector size: %d elements (%.2f MB per array)\n", numElements, size / (1024.0f * 1024.0f));

    // Calculate split: divide work between 2 GPUs
    int elementsPerGPU = numElements / 2;
    int remainder = numElements % 2;
    
    int numElements_GPU0 = elementsPerGPU + remainder;  // Give remainder to GPU 0
    int numElements_GPU1 = elementsPerGPU;
    
    size_t size_GPU0 = numElements_GPU0 * sizeof(float);
    size_t size_GPU1 = numElements_GPU1 * sizeof(float);
    
    printf("GPU 0 will process: %d elements (%.2f MB)\n", numElements_GPU0, size_GPU0 / (1024.0f * 1024.0f));
    printf("GPU 1 will process: %d elements (%.2f MB)\n", numElements_GPU1, size_GPU1 / (1024.0f * 1024.0f));

    hipStream_t stream0, stream1; // Declare streams for asynchronous operations
    float *h_A = NULL, *h_B = NULL, *h_C = NULL; // Allocate host memory
    float *d_A0 = NULL, *d_B0 = NULL, *d_C0 = NULL; // Device pointers for GPU 0
    float *d_A1 = NULL, *d_B1 = NULL, *d_C1 = NULL; // Device pointers for GPU 1

    // ============== setup host side ==============
    allocateHostMemory(&h_A, size, "A");
    allocateHostMemory(&h_B, size, "B");
    allocateHostMemory(&h_C, size, "C");

    // Initialize host arrays
    printf("\nInitializing input vectors...\n");
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // ============== GPU 0 Setup ==============
    printf("\n=== Setting up GPU 0 ===\n");
    setDevice(0);

    // Create stream for GPU 0
    createStream(&stream0, 0);

    // Allocate memory on GPU 0
    allocateDeviceMemory(&d_A0, size_GPU0, "A", 0);
    allocateDeviceMemory(&d_B0, size_GPU0, "B", 0);
    allocateDeviceMemory(&d_C0, size_GPU0, "C", 0);

    printf("Allocated %.2f MB on GPU 0\n", (size_GPU0 * 3) / (1024.0f * 1024.0f));

    // Copy data to GPU 0 (asynchronously)
    copyMemoryAsync(d_A0, h_A, size_GPU0, hipMemcpyHostToDevice, stream0, "copy data to", 0);
    copyMemoryAsync(d_B0, h_B, size_GPU0, hipMemcpyHostToDevice, stream0, "copy data to", 0);

    // ============== GPU 1 Setup ==============
    printf("\n=== Setting up GPU 1 ===\n");
    setDevice(1);

    // Create stream for GPU 1
    createStream(&stream1, 1);

    
    // Allocate memory on GPU 1
    allocateDeviceMemory(&d_A1, size_GPU1, "A", 1);
    allocateDeviceMemory(&d_B1, size_GPU1, "B", 1);
    allocateDeviceMemory(&d_C1, size_GPU1, "C", 1);

    printf("Allocated %.2f MB on GPU 1\n", (size_GPU1 * 3) / (1024.0f * 1024.0f));

    // Copy data to GPU 1 (asynchronously) - offset to second half of data
    copyMemoryAsync(d_A1, h_A + numElements_GPU0, size_GPU1, hipMemcpyHostToDevice, stream1, "copy data to", 1);
    copyMemoryAsync(d_B1, h_B + numElements_GPU0, size_GPU1, hipMemcpyHostToDevice, stream1, "copy data to", 1);

    // ============== Launch Kernels on Both GPUs ==============
    printf("\n=== Launching kernels on both GPUs ===\n");
    
    int threadsPerBlock = 256;

    // Launch kernel on GPU 0
    setDevice(0);
    launchKernel(d_A0, d_B0, d_C0, numElements_GPU0, threadsPerBlock, stream0, 0);

    // Launch kernel on GPU 1
    setDevice(1);
    launchKernel(d_A1, d_B1, d_C1, numElements_GPU1, threadsPerBlock, stream1, 1);

    // Wait for both GPUs to finish
    synchronizeStream(stream0, 0);
    synchronizeStream(stream1, 1);


    // ============== Copy Results Back ==============
    printf("\n=== Copying results back to host ===\n");
    
    // Copy results from GPU 0 (asynchronously)
    setDevice(0);
    copyMemoryAsync(h_C, d_C0, size_GPU0, hipMemcpyDeviceToHost, stream0, "copy results from", 0);
    
    // Copy results from GPU 1 (asynchronously) - offset to second half
    setDevice(1);
    copyMemoryAsync(h_C + numElements_GPU0, d_C1, size_GPU1, hipMemcpyDeviceToHost, stream1, "copy results from", 1);
    
    // ============== Verify Results ==============
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
        
    // Free GPU 0 memory
    setDevice(0);
    freeDeviceMemory(d_A0, "A", 0);
    freeDeviceMemory(d_B0, "B", 0);
    freeDeviceMemory(d_C0, "C", 0);
    
    // Free GPU 1 memory
    setDevice(1);
    freeDeviceMemory(d_A1, "A", 1);
    freeDeviceMemory(d_B1, "B", 1);
    freeDeviceMemory(d_C1, "C", 1);

    // Destroy streams
    destroyStream(stream0, 0);
    destroyStream(stream1, 1);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return passed ? 0 : 1;
}

