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

// Helper function to allocate device memory with error checking
void allocateDeviceMemory(float **d_ptr, size_t size, const char *vectorName)
{
    hipError_t err = hipMalloc((void **)d_ptr, size);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector %s (error code %s)!\n", 
                vectorName, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to copy memory with error checking
void copyMemory(void *dst, const void *src, size_t size, hipMemcpyKind kind, const char *vectorName, const char *direction)
{
    hipError_t err = hipMemcpy(dst, src, size, kind);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector %s %s (error code %s)!\n", 
                vectorName, direction, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Helper function to free device memory with error checking
void freeDeviceMemory(void *d_ptr, const char *vectorName)
{
    hipError_t err = hipFree(d_ptr);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector %s (error code %s)!\n", 
                vectorName, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    // Print header
    printf("[Vector Addition using AMD GPU with HIP]\n");

    // Error code to check return values for HIP API calls
    hipError_t err = hipSuccess;

    // Number of elements in arrays
    int numElements = 100000000;
    size_t size = numElements * sizeof(float);
    printf("Vector size: %d elements\n", numElements);

    // Allocate host memory
    float *h_A = NULL;
    allocateHostMemory(&h_A, size, "A");
    
    float *h_B = NULL;
    allocateHostMemory(&h_B, size, "B");
    
    float *h_C = NULL;
    allocateHostMemory(&h_C, size, "C");

    // Initialize host arrays
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A = NULL;
    allocateDeviceMemory(&d_A, size, "A");

    float *d_B = NULL;
    allocateDeviceMemory(&d_B, size, "B");

    float *d_C = NULL;
    allocateDeviceMemory(&d_C, size, "C");

    // Copy input vectors from host to device
    printf("Copying input data from host to device...\n");
    copyMemory(d_A, h_A, size, hipMemcpyHostToDevice, "A", "from host to device");
    copyMemory(d_B, h_B, size, hipMemcpyHostToDevice, "B", "from host to device");

    // Launch the vector addition kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, numElements);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    printf("Copying output data from device to host...\n");
    copyMemory(h_C, d_C, size, hipMemcpyDeviceToHost, "C", "from device to host");

    // Verify that the result vector is correct
    printf("Verifying results...\n");
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device memory
    freeDeviceMemory(d_A, "A");

    freeDeviceMemory(d_B, "B");

    freeDeviceMemory(d_C, "C");

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
