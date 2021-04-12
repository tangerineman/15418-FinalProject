#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <string>

#include "simRenderer.h"
#include "sceneLoader.h"

#include "image.h"

struct GlobalConstants {
  Benchmark benchmark;

  int numberOfParticles;

  float* position;
  float* velocity;
  float* color;

  int imageWidth;
  int imageHeight;
  float* imageData;

  int* locks;
};

__constant__ GlobalConstants cuConstRendererParams;

__global__ void kernelClearImage(float r, float g, float b, float a) {
  int imageX = blockIdx.x * blockDim.x + threadIdx.x;
  int imageY = blockIdx.y * blockDim.y + threadIdx.y;

  int width = cuConstRendererParams.imageWidth;
  int height = cuConstRendererParams.imageHeight;

  if (imageX >= width || imageY >= height)
      return;

  int offset = 4 * (imageY * width + imageX);
  float4 value = make_float4(r, g, b, a);

  // Write to global memory: As an optimization, this code uses a float4
  // store, which results in more efficient code than if it were coded as
  // four separate float stores.
  *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

__global__ void kernelBasic() {
  const float dt = 1.f / 60.f;

  float* velocity = cuConstRendererParams.velocity;
  float* position = cuConstRendererParams.position;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cuConstRendererParams.numberOfParticles || index < 0)
      return;

  int pIdx = index * 2;
  int vIdx = index * 2;

  position[pIdx] += velocity[vIdx] * dt;
  position[pIdx+1] += velocity[vIdx+1] * dt;

  //vel stays the same
}

__global__ void kernelRenderParticles() {
  int index = blockIdx.x * blockDim.x + threadIdx.x;


  if (index >= cuConstRendererParams.numberOfParticles)
      return;

  //int index2 = 2 * index;
  //int index4 = 4 * index;

  float2 p = *(float2*)(&cuConstRendererParams.position[index]);

  int px = ceil(p.x);
  int py = ceil(p.y);

  float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (py * cuConstRendererParams.imageWidth + px)]);

  *imgPtr = ((float4*)(&cuConstRendererParams.color))[index];
}


void SimRenderer::clearImage() {
    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
      (image->width + blockDim.x - 1) / blockDim.x,
      (image->height + blockDim.y - 1) / blockDim.y);

    kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);

    cudaDeviceSynchronize();
}

SimRenderer::SimRenderer() {
  image = NULL;
  benchmark = SIMPLE;

  numberOfParticles = 0;

  cudaDevicePosition = NULL;
  cudaDeviceVelocity = NULL;
  cudaDeviceColor = NULL;
  cudaDeviceImageData = NULL;
}


SimRenderer::~SimRenderer() {
  if (image) {
    delete image;
  }

  if (position) {
    delete [] position;
    delete [] velocity;
    delete [] color;
  }

  if (cudaDevicePosition) {
    cudaFree(cudaDevicePosition);
    cudaFree(cudaDeviceVelocity);
    cudaFree(cudaDeviceColor);
    cudaFree(cudaDeviceImageData);
  }
}

const Image*
SimRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
SimRenderer::loadScene(Benchmark bm) {
  loadParticleScene(bm, numberOfParticles, position, velocity, color);
}

void
SimRenderer::setup() {
  int deviceCount = 0;
  bool isFastGPU = false;
  std::string name;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Initializing CUDA for SimRenderer\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i=0; i<deviceCount; i++) {
      cudaDeviceProp deviceProps;
      cudaGetDeviceProperties(&deviceProps, i);
      name = deviceProps.name;
      if (name.compare("GeForce RTX 2080") == 0)
      {
          isFastGPU = true;
      }

      printf("Device %d: %s\n", i, deviceProps.name);
      printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
      printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
      printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
  if (!isFastGPU)
  {
      printf("WARNING: "
             "You're not running on a fast GPU, please consider using "
             "NVIDIA RTX 2080.\n");
      printf("---------------------------------------------------------\n");
  }

  // By this time the scene should be loaded.  Now copy all the key
  // data structures into device memory so they are accessible to
  // CUDA kernels
  //
  // See the CUDA Programmer's Guide for descriptions of
  // cudaMalloc and cudaMemcpy

  cudaMalloc(&cudaDevicePosition, sizeof(float) * 2 * numberOfParticles);
  cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfParticles);
  cudaMalloc(&cudaDeviceColor, sizeof(float) * 4 * numberOfParticles);
  cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

  cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfParticles, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfParticles, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 4 * numberOfParticles, cudaMemcpyHostToDevice);


  // Initialize parameters in constant memory.  We didn't talk about
  // constant memory in class, but the use of read-only constant
  // memory here is an optimization over just sticking these values
  // in device global memory.  NVIDIA GPUs have a few special tricks
  // for optimizing access to constant memory.  Using global memory
  // here would have worked just as well.  See the Programmer's
  // Guide for more information about constant memory.

  GlobalConstants params;
  params.benchmark = benchmark;
  params.numberOfParticles = numberOfParticles;
  params.imageWidth = image->width;
  params.imageHeight = image->height;
  params.position = cudaDevicePosition;
  params.velocity = cudaDeviceVelocity;
  params.color = cudaDeviceColor;
  params.imageData = cudaDeviceImageData;

  cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));
  /*
  // Also need to copy over the noise lookup tables, so we can
  // implement noise on the GPU
  int* permX;
  int* permY;
  float* value1D;
  getNoiseTables(&permX, &permY, &value1D);
  cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
  cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
  cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);
  */
}

void
SimRenderer::advanceAnimation() {
   // 256 threads per block is a healthy number
  dim3 blockDim(256, 1);
  dim3 gridDim((numberOfParticles + blockDim.x - 1) / blockDim.x);

  kernelBasic<<<gridDim, blockDim>>>();

  cudaDeviceSynchronize();
}

void
SimRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

void
SimRenderer::render() {
  // 256 threads per block is a healthy number
  dim3 blockDim(16, 16);
  dim3 gridDim(16, 16);

  kernelRenderParticles<<<gridDim, blockDim>>>();
  cudaDeviceSynchronize();
}
