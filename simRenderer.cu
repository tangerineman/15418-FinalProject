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

  int currX = ceil(position[pIdx]);
  int currY = ceil(position[pIdx + 1]);

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

  //int h = cuConstRendererParams.imageHeight;
  int w = cuConstRendererParams.imageWidth;

  int index = row * w + col;

  if(newVelField[index].x == 200.f) {
    printf("row, col for 200: %d, %d\n", row, col);
  }
  cuConstRendererParams.velField[2*index] = newVelField[index].x;
  cuConstRendererParams.velField[2*index+1] = newVelField[index].y;
}

__global__ void kernelVecMomentum(float2* newVelField) {
  uint col = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint row = (blockIdx.y * blockDim.y) + threadIdx.y;

  int w = cuConstRendererParams.imageWidth;
  int h = cuConstRendererParams.imageHeight;

  int index = row * w + col;

  float2* velField2 = (float2*) cuConstRendererParams.velField;

  float2 curr = velField2[index];
  float2 sum = make_float2(curr.x, curr.y);

  float2 zero = make_float2(0.f, 0.f);

  float2 topLeft = (row == 0 || col == 0) ? zero : velField2[index - w - 1];
  if(topLeft.y == 0.f)
    topLeft.y = 0.0000001f;
  float2 top = (row == 0) ? zero : velField2[index - w];
  if(top.y == 0.f)
    top.y = 0.0000001f;
  float2 topRight = (row == 0 || col == (w-1)) ? zero : velField2[index - w + 1];
  if(topRight.y == 0.f)
    topRight.y = 0.0000001f;
  float2 left = (col == 0) ? zero : velField2[index - 1];
  if(left.y == 0.f)
    left.y = 0.0000001f;
  float2 right = (col == (w-1)) ? zero : velField2[index + 1];
  if(right.y == 0.f)
    right.y = 0.0000001f;
  float2 botLeft = (row == (h-1) || col == 0) ? zero : velField2[index + w - 1];
  if(botLeft.y == 0.f)
    botLeft.y = 0.0000001f;
  float2 bot = (row == (h-1)) ? zero : velField2[index + w];
  if(bot.y == 0.f)
    bot.y = 0.0000001f;
  float2 botRight = (row == (h-1) || col == (w-1)) ? zero : velField2[index + w + 1];
  if(botRight.y == 0.f)
    botRight.y = 0.0000001f;

  float counter = 1.f;
  float atres;

  atres = atan(topLeft.y / topLeft.x);
  if(topLeft.x > 0.f && topLeft.y > 0.f && atres > 0.523f && atres < 1.05f) {
    counter += 1.f;
    sum = f2add(sum, topLeft);
  }

  atres = atan(top.y / top.x);
  if(top.y > 0.f && atres > 1.05f && atres < 2.09f) {
    counter += 1.f;
    sum = f2add(sum, top);
  }

  atres = abs(atan(topRight.y / topRight.x));
  if(topRight.x < 0.f && topRight.y > 0.f && atres > 0.523f && atres < 1.05f) {
    counter += 1.f;
    sum = f2add(sum, topRight);
  }

  atres = atan(left.y / left.x);
  if(left.x > 0.f && atres < 1.05f && atres > -1.05f) {
    counter += 1.f;
    sum = f2add(sum, left);
  }

  atres = atan(right.y / right.x);
  if(right.x < 0.f && atres < 1.05f && atres > -1.05f) {
    counter += 1.f;
    sum = f2add(sum, right);
  }

  atres = abs(atan(botLeft.y / botLeft.x));
  if(botLeft.y < 0.f && botLeft.x > 0.f && atres > 0.523f && atres < 1.05f) {
    counter += 1.f;
    sum = f2add(sum, botLeft);
  }

  atres = abs(atan(bot.y / bot.x));
  if(bot.y < 0.f && atres > 1.05f && atres < 2.09f) {
    counter += 1.f;
    sum = f2add(sum, bot);
  }

  atres = atan(botRight.y / botRight.x);
  if(botRight.x < 0.f && botRight.y < 0.f && atres > 0.523f && atres < 1.05f) {
    counter += 1.f;
    sum = f2add(sum, botRight);
  }

  //if(curr.x != 0.f && curr.y != 0.f) {
  //  printf("curr.x, y: %f, %f, new x, y: %f, %f\n", curr.x, curr.y, sum.x, sum.y);
  //}
  //printf("sumx, y; %f, %f, counter: %d\n", sum.x, sum.y, counter);

  sum.x = counter == 0.f ? 0.f : sum.x / counter;
  sum.y = counter == 0.f ? 0.f : sum.y / counter;
  //printf("curr.x, y: %f, %f, new x, y: %f, %f\n", curr.x, curr.y, sum.x, sum.y);


  newVelField[index] = sum;
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

  newVel.x = curr.x + (topRight.x + topRight.y + botLeft.x + botLeft.y +
                       topLeft.x - topLeft.y + botRight.x - botRight.y +
                       + 2 * (left.x + right.x - top.x - bot.x)
                       - 4 * curr.x)/8.f;

   newVel.y = curr.y + (topLeft.x + topLeft.y + botRight.x + botRight.y +
                        topRight.y - topRight.x + botLeft.y - botLeft.x +
                        + 2 * (top.y + bot.y - right.y - left.y)
                        - 4 * curr.y)/8.f;
  /*
  newVel.x =  curr.x +
              (dp(f2add(topLeft, botRight), make_float2(1.f, 1.f)) +
               dp(f2add(botLeft, topRight), make_float2(1.f, -1.f)) +
               2 * dp(f2sub(f2add(left, right), f2add(top, bot)), make_float2(2.f, -2.f)) +
               curr.x * -4.f) / 8.f;
  newVel.y = curr.y +
             (dp(f2add(topLeft, botRight), make_float2(1.f, 1.f)) -
              dp(f2add(botLeft, topRight), make_float2(1.f, -1.f)) -
              -2 * dp(f2sub(f2add(left, right), f2add(top, bot)), make_float2(2.f, -2.f)) +
              curr.y * -4.f) / 8.f;*/
  /*
  if(row == 258 && curr.x != 0) {
    printf("\ncurr: %f, %f\ntopLeft: %f, %f\ntop: %f, %f\ntopright: %f, %f\nleft: %f, %f\n, right: %f, %f\nbotleft: %f, %f\nbot: %f, %f\nbotRight: %f, %f\nnew: %f, %f\n",
    curr.x, curr.y, topLeft.x, topLeft.y,
    top.x, top.y, topRight.x, topRight.y,
    left.x, left.y, right.x, right.y,
    botLeft.x, botLeft.y, bot.x, bot.y,
    botRight.x, botRight.y, newVel.x, newVel.y);

  }*/
/*
  if(newVel.x == 200.f)
    printf("row for entering 200: %d, %d\n", row, col);
  if(row == 253) {
    printf("newvel (x, y): %f, %f\nprev was : %f, %f", newVel.x, newVel.y, newVelField[index].x, newVelField[index].y);
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
  printf("IN RENDERER\n");
  // 256 threads per block is a healthy number
  dim3 blockDimVec(8, 8);
  dim3 gridDimVec(64, 64);
  dim3 blockDimParticles(256);
  dim3 gridDimParticles(16, 16);

  float2* cudaDeviceVelFieldUpdated;

  cudaMalloc(&cudaDeviceVelFieldUpdated, sizeof(float) * 2 * image->width * image->height);

  printf("IN RENDERER P2\n");

  kernelVecMomentum<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
  cudaDeviceSynchronize();
  kernelVelFieldCopy<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
  cudaDeviceSynchronize();

  for(int i = 0; i < 1; i++) {
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
