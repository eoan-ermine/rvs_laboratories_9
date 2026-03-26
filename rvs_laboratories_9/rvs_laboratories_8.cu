#include "wb.h"

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define value(arr, i, j, k) arr[((i) * width + (j)) * depth + (k)]
#define in(i, j, k) value(input, i, j, k)
#define out(i, j, k) value(output, i, j, k)

__global__ void stencil(float *output, float *input, int width, int height,
                        int depth) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i <= 0 || i >= height - 1 ||
      j <= 0 || j >= width - 1 ||
      k <= 0 || k >= depth - 1) {
    return;
  }

  float res = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) +
              in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) -
              6 * in(i, j, k);
  out(i, j, k) = max(min(res, 1.0f), 0.0f);
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData,
                           int width, int height, int depth) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y,
               (depth + blockDim.z - 1) / blockDim.z);

  stencil<<<gridDim, blockDim>>>(deviceOutputData, deviceInputData,
                                 width, height, depth);
  cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  arg = wbArg_read(argc, argv);

  inputFile = wbArg_getInputFile(arg, 0);

  input = wbImport(inputFile);

  width = wbImage_getWidth(input);
  height = wbImage_getHeight(input);
  depth = wbImage_getChannels(input);

  output = wbImage_new(width, height, depth);

  hostInputData = wbImage_getData(input);
  hostOutputData = wbImage_getData(output);

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
  cudaMalloc((void **)&deviceOutputData,
             width * height * depth * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData,
             width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputData, deviceOutputData,
             width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbSolution(arg, output);

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  wbImage_delete(output);
  wbImage_delete(input);

  return 0;
}