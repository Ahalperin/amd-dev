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
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A = NULL;
    err = hipMalloc((void **)&d_A, size);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = hipMalloc((void **)&d_B, size);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_C = NULL;
    err = hipMalloc((void **)&d_C, size);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy input vectors from host to device
    printf("Copying input data from host to device...\n");
    err = hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the vector addition kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_B, numElements);

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    printf("Copying output data from device to host...\n");
    err = hipMemcpy(h_C, d_B, size, hipMemcpyDeviceToHost);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

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
    err = hipFree(d_A);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = hipFree(d_B);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = hipFree(d_C);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

