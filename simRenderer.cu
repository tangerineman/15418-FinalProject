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

  float* velField;

  int numberOfParticles;
  float* position;
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

  float* velField = cuConstRendererParams.velField;
  float* position = cuConstRendererParams.position;

  int w = cuConstRendererParams.imageWidth;
  int h = cuConstRendererParams.imageHeight;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cuConstRendererParams.numberOfParticles || index < 0)
      return;

  int pIdx = index * 2;

  int currX = position[pIdx];
  int currY = position[pIdx + 1];

  position[pIdx] += velField[2*(w * currY + currX)] * dt;
  if(ceil(position[pIdx]) <= 0)
    position[pIdx] = 0;
  else if (ceil(position[pIdx]) >= w-1)
    position[pIdx] = w-1;
  /*
  if(ceil(position[pIdx]) <= 0 || ceil(position[pIdx]) >= w-1) {
    position[pIdx] = (ceil(position[pIdx]) <= 0) ? 0.f : (float) w-1;
    velocity[pIdx] *= -1.f;
  }*/
  position[pIdx+1] += velField[2*(w * currY + currX) + 1] * dt;
  if(ceil(position[pIdx+1]) <= 0)
    position[pIdx+1] = 0;
  else if (ceil(position[pIdx+1]) >= h-1)
    position[pIdx+1] = h-1;
  /*
  if(ceil(position[pIdx+1]) <= 0 || ceil(position[pIdx+1]) >= h-1) {
    velocity[pIdx+1] *= -1.f;
    position[pIdx+1] = (ceil(position[pIdx+1]) <= 0) ? 0.f : (float) h-1;
  }*/

  //vel stays the same
}

__device__ float dp(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}

__device__ float2 f2add(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 f2sub(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

__global__ void kernelVelFieldCopy(float2* newVelField) {
  uint col = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint row = (blockIdx.y * blockDim.y) + threadIdx.y;

  int h = cuConstRendererParams.imageHeight;
  int w = cuConstRendererParams.imageWidth;

  int index = row * w + col;

  cuConstRendererParams.velField[2*index] = newVelField[index].x;
  cuConstRendererParams.velField[2*index+1] = newVelField[index].y;
}

__global__ void kernelUpdateVectorField(float2* newVelField) {
  uint col = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint row = (blockIdx.y * blockDim.y) + threadIdx.y;

  int h = cuConstRendererParams.imageHeight;
  int w = cuConstRendererParams.imageWidth;

  int index = row * w + col;

  float2* velField2 = (float2*) cuConstRendererParams.velField;

  float2 curr = velField2[index];

  float2 zero = make_float2(0.f, 0.f);

  float2 topLeft = (row == 0 || col == 0) ? zero : velField2[index - w - 1];
  float2 top = (row == 0) ? zero : velField2[index - w];
  float2 topRight = (row == 0 || col == (w-1)) ? zero : velField2[index - w + 1];
  float2 left = (col == 0) ? zero : velField2[index - 1];
  float2 right = (col == (w-1)) ? zero : velField2[index + 1];
  float2 botLeft = (row == (h-1) || col == 0) ? zero : velField2[index + w - 1];
  float2 bot = (row == (h-1)) ? zero : velField2[index + w];
  float2 botRight = (row == (h-1) || col == (w-1)) ? zero : velField2[index + w + 1];

  float2 newVel;

  newVel.x =  curr.x +
              (dp(f2add(topLeft, botRight), make_float2(1.f, 1.f)) +
               dp(f2add(botLeft, topRight), make_float2(1.f, -1.f)) +
               2 * dp(f2sub(f2add(left, right), f2add(top, bot)), make_float2(2.f, -2.f)) +
               curr.x * -4.f) / 8.f;
  newVel.y = curr.y +
             (dp(f2add(topLeft, botRight), make_float2(1.f, 1.f)) -
              dp(f2add(botLeft, topRight), make_float2(1.f, -1.f)) -
              -2 * dp(f2sub(f2add(left, right), f2add(top, bot)), make_float2(2.f, -2.f)) +
              curr.y * -4.f) / 8.f;


  //curr.y + (dp(f2add(topLeft, botRight), make_float2(1.f, 1.f)) - dp(f2add(botLeft, topRight), make_float2(1.f, -1.f)) + -2.f * (left.y + right.y - top.y - bot.y) + curr.y * -4.f) / 8.f;
  /*
  if(row == 0) {
    newVelField[index] = make_float2(0.f, 10.f);
  } else if (col == 0) {
    newVelField[index] = make_float2(10.f, 0.f);
  } else if (row == h-1) {
    newVelField[index] = make_float2(0.f, -10.f);
  } else if (col == w-1) {
    newVelField[index] = make_float2(-10.f, 0.f);
  } else {
    newVelField[index] = newVel;
  }*/
  newVelField[index] = newVel;
}

__global__ void kernelRenderParticles() {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= cuConstRendererParams.numberOfParticles) {
      //printf("HOW\n");
      return;
  }
  //int index2 = 2 * index;
  //int index4 = 4 * index;
  //printf("oof1\n");
  float2 p = ((float2*)cuConstRendererParams.position)[index];
  int px = ceil(p.x);
  int py = ceil(p.y);


  float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (py * cuConstRendererParams.imageWidth + px)]);

  float4 oof = ((float4*)cuConstRendererParams.color)[index];

  *imgPtr = make_float4(oof.x, oof.y, oof.z, oof.w);//((float4*)(&cuConstRendererParams.color))[index];
}


void SimRenderer::clearImage() {
    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
      (image->width + blockDim.x - 1) / blockDim.x,
      (image->height + blockDim.y - 1) / blockDim.y);

    kernelClearImage<<<gridDim, blockDim>>>(0.f, 0.f, 0.f, 1.f);

    cudaDeviceSynchronize();
}

SimRenderer::SimRenderer() {
  image = NULL;
  benchmark = STREAM1;

  numberOfParticles = 0;

  cudaDevicePosition = NULL;
  cudaDeviceVelField = NULL;
  cudaDeviceColor = NULL;
  cudaDeviceImageData = NULL;
}


SimRenderer::~SimRenderer() {
  if (image) {
    delete image;
  }

  if (position) {
    delete [] position;
    delete [] velField;
    delete [] color;
  }

  if (cudaDevicePosition) {
    cudaFree(cudaDevicePosition);
    cudaFree(cudaDeviceVelField);
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
    printf("copied\n");
    return image;
}

void
SimRenderer::loadScene(Benchmark bm) {
  loadParticleScene(bm, image->width, image->height, numberOfParticles, position, velField, color);
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
  cudaMalloc(&cudaDeviceVelField, sizeof(float) * 2 * image->width * image->height);
  cudaMalloc(&cudaDeviceColor, sizeof(float) * 4 * numberOfParticles);
  cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

  cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 2 * numberOfParticles, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceVelField, velField, sizeof(float) * 2 * image->width * image->height, cudaMemcpyHostToDevice);
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
  params.velField = cudaDeviceVelField;
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
  dim3 blockDimVec(8, 8);
  dim3 gridDimVec(64, 64);
  dim3 blockDimParticles(256);
  dim3 gridDimParticles(16, 16);

  float2* cudaDeviceVelFieldUpdated;

  cudaMalloc(&cudaDeviceVelFieldUpdated, sizeof(float) * 2 * image->width * image->height);

  for(int i = 0; i < 30; i++) {
    kernelUpdateVectorField<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
    cudaDeviceSynchronize();
    kernelVelFieldCopy<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
    cudaDeviceSynchronize();
    //cudaMemcpy(cuConstRendererParams.velField, cudaDeviceVelFieldUpdated, sizeof(float) * 2 * image->width * image->height, cudaMemcpyDeviceToDevice);
  }

  cudaFree(cudaDeviceVelFieldUpdated);

  cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

  kernelRenderParticles<<<gridDimParticles, blockDimParticles>>>();
  //cudaDeviceSynchronize();
  cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
}
