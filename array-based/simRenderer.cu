#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <string>

#include "simRenderer.h"
#include "sceneLoader.h"

#include "image.h"

#define ITERS 50

struct GlobalConstants {
  Benchmark benchmark;

  float* velField;
  float* position;
  float* colors;
  float* spawners;

  int numParticles;
  int maxNumParticles;
  int numSpawners;

  int currParticleIndex;
  int currParticleLast;

  int imageWidth;
  int imageHeight;
  float* imageData;

};

__constant__ GlobalConstants cuConstRendererParams;
//unsigned int MAX_PARTICLES = 2048;

// linked list kernels
////////////////////////////////////////////////////////////////////////////////

// helper function to quickly pick a color given an id
__device__ float *pickColor(int id){
  float *color = (float*)malloc(sizeof(float) * 4);
  if(id % 4 == 0) {
    color[0] = 1.f;
    color[1] = 0.f;
    color[2] = 0.f;
    color[3] = 1.f;
  }
  if(id % 4 == 1) {
    color[0] = 0.f;
    color[1] = 1.f;
    color[2] = 0.f;
    color[3] = 1.f;
  }
  if(id % 4 == 2) {
    color[0] = 0.f;
    color[1] = 0.f;
    color[2] = 1.f;
    color[3] = 1.f;
  }
  if(id % 4 == 3) {
    color[0] = 0.f;
    color[1] = .5;
    color[2] = .5;
    color[3] = 1.f;
  }

  return color;
}

////////////////////////////////////////////////////////////////////////////////

// kernel to clear an image with black pixels
__global__ void kernelClearImage(float r, float g, float b, float a) {
  int imageX = blockIdx.x * blockDim.x + threadIdx.x;
  int imageY = blockIdx.y * blockDim.y + threadIdx.y;

  int width = cuConstRendererParams.imageWidth;
  int height = cuConstRendererParams.imageHeight;

  if (imageX >= width || imageY >= height)
      return;

  int offset = 4 * (imageY * width + imageX);
  float4 value = make_float4(r, g, b, a);

  *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

//kernel to update a particles positions acording to the corresponding velocity
// at the particles position
__global__ void kernelBasic(int currParticleIndex, int currParticleLast) {
  const float dt = 1.f / 60.f;

  float* velField = cuConstRendererParams.velField;
  float* position = cuConstRendererParams.position;

  //int currParticleIndex = cuConstRendererParams.currParticleIndex;
  int maxNumParticles = cuConstRendererParams.maxNumParticles;

  int w = cuConstRendererParams.imageWidth;
  int h = cuConstRendererParams.imageHeight;

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  index = (index + currParticleIndex) % maxNumParticles;

  if(!(index >= currParticleIndex && index < currParticleLast))
    return;

  int pIdx = index * 2;

  int currX = ceil(position[pIdx]);
  int currY = ceil(position[pIdx + 1]);

  //update x position
  position[pIdx] += velField[2*(w * currY + currX)] * dt;

  //boundaries to ensure particles do not go off screen
  if(ceil(position[pIdx]) <= 0)
    position[pIdx] = 0;
  else if (ceil(position[pIdx]) >= w-1)
    position[pIdx] = w-1;

  //update y position
  position[pIdx+1] += velField[2*(w * currY + currX) + 1] * dt;
  //boundaries to ensure particles do not go off screen
  if(ceil(position[pIdx+1]) <= 0)
    position[pIdx+1] = 0;
  else if (ceil(position[pIdx+1]) >= h-1)
    position[pIdx+1] = h-1;

}

//helper function to add two float2's
__device__ float2 f2add(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

//kernel to copy vel field data from input to cuda mem
__global__ void kernelVelFieldCopy(float2* newVelField) {
  uint col = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint row = (blockIdx.y * blockDim.y) + threadIdx.y;

//   int h = cuConstRendererParams.imageHeight;
  int w = cuConstRendererParams.imageWidth;

  int index = row * w + col;

  cuConstRendererParams.velField[2*index] = newVelField[index].x;
  cuConstRendererParams.velField[2*index+1] = newVelField[index].y;
}

//kernel to update the velocity based on momentum of the field
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

    // calculate surrounding pixels and ensure they are not 0 for division issues
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

    // the following is logic using angles to determine if the surrounding pixels vectors
    // intersect with the center pixel being computed
    atres = abs(atan(topLeft.y / topLeft.x));
    if(topLeft.x > 0.f && topLeft.y > 0.f && atres > 0.523f && atres < 1.05f) {
      counter += 1.f;
      sum = f2add(sum, topLeft);
    }

    atres = abs(atan(top.y / top.x));
    if(top.y > 0.f && atres >= 1.05f && atres <= 2.09f) {
      counter += 1.f;
      sum = f2add(sum, top);
    }

    atres = abs(atan(topRight.y / topRight.x));
    if(topRight.x < 0.f && topRight.y > 0.f && atres > 0.523f && atres < 1.05f) {
      counter += 1.f;
      sum = f2add(sum, topRight);
    }

    atres = abs(atan(left.y / left.x));
    if(left.x > 0.f && atres < 1.05f) {
      counter += 1.f;
      sum = f2add(sum, left);
    }

    atres = abs(atan(right.y / right.x));
    if(right.x < 0.f && atres < 1.05f) {
      counter += 1.f;
      sum = f2add(sum, right);
    }

    atres = abs(atan(botLeft.y / botLeft.x));
    if(botLeft.y < 0.f && botLeft.x > 0.f && atres > 0.523f && atres < 1.05f) {
      counter += 1.f;
      sum = f2add(sum, botLeft);
    }

    atres = abs(atan(bot.y / bot.x));
    if(bot.y < 0.f && atres >= 1.05f && atres <= 2.09f) {
      counter += 1.f;
      sum = f2add(sum, bot);
    }

    atres = atan(botRight.y / botRight.x);
    if(botRight.x < 0.f && botRight.y < 0.f && atres > 0.523f && atres < 1.05f) {
      counter += 1.f;
      sum = f2add(sum, botRight);
    }

    sum.x = counter == 0.f ? 0.f : sum.x / counter;
    sum.y = counter == 0.f ? 0.f : sum.y / counter;

    newVelField[index] = sum;
  }

// kernel to remove the divergence from the vector field
__global__ void kernelUpdateVectorField(float2* newVelField) {
    uint col = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint row = (blockIdx.y * blockDim.y) + threadIdx.y;

    int h = cuConstRendererParams.imageHeight;
    int w = cuConstRendererParams.imageWidth;

    if(col >= w || row >= h) {
      return;
    }

    int index = row * w + col;

    float2* velField2 = (float2*) cuConstRendererParams.velField;

    float2 curr = velField2[index];

    float2 zero = make_float2(0.f, 0.f);

    // surrounding pixel logic
    float2 botLeft = (row == 0 || col == 0) ? zero : velField2[index - w - 1];
    float2 bot = (row == 0) ? zero : velField2[index - w];
    float2 botRight = (row == 0 || col == (w-1)) ? zero : velField2[index - w + 1];
    float2 left = (col == 0) ? zero : velField2[index - 1];
    float2 right = (col == (w-1)) ? zero : velField2[index + 1];
    float2 topLeft = (row == (h-1) || col == 0) ? zero : velField2[index + w - 1];
    float2 top = (row == (h-1)) ? zero : velField2[index + w];
    float2 topRight = (row == (h-1) || col == (w-1)) ? zero : velField2[index + w + 1];

    float2 newVel;

    // calculate gradient in x direction and add it to current x val
    newVel.x = curr.x + (topRight.x + topRight.y + botLeft.x + botLeft.y +
                         topLeft.x - topLeft.y + botRight.x - botRight.y +
                         + 2 * (left.x + right.x - top.x - bot.x)
                         - 4 * curr.x)/8.f;
     // calculate gradient in y direction and add it to current y val
     newVel.y = curr.y + (topRight.x + topRight.y + botLeft.x + botLeft.y +
                          topLeft.y - topLeft.x + botRight.y - botRight.x +
                          + 2 * (top.y + bot.y - right.y - left.y)
                          - 4 * curr.y)/8.f;

    newVelField[index] = newVel;
}

// kernel to spawn particles from spawning locations
__global__ void kernelSpawnParticles(int currParticleIndex, int currParticleLast) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int cpi = currParticleIndex;
  int cpl = currParticleLast;

  //int currParticleIndex = cuConstRendererParams.currParticleIndex;
  int maxNumParticles = cuConstRendererParams.maxNumParticles;
  int numSpawners = cuConstRendererParams.numSpawners;


  int posIndex = (index + cpl) % maxNumParticles;

  if(cpl < cpi && !(posIndex < cpi && posIndex >= cpl)) {
    return;
  }
  if (cpl > cpi && posIndex < cpl && posIndex >= cpi) {
    return;
  }

  float2 s = ((float2*)cuConstRendererParams.spawners)[index];
  float2* p = &((float2*)cuConstRendererParams.position)[posIndex];
  float4* c = &((float4*)cuConstRendererParams.colors)[posIndex];

  *p = s;

  float* newCol = pickColor(2);
  *c = make_float4(newCol[0], newCol[1], newCol[2], newCol[3]);

}

// kernel to move the particles from their positions and update the image with the appropriate color
__global__ void kernelRenderParticles(int currParticleIndex, int currParticleLast) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //int currParticleIndex = cuConstRendererParams.currParticleIndex;
  int maxNumParticles = cuConstRendererParams.maxNumParticles;

  index = (index + currParticleIndex) % maxNumParticles;

  if((currParticleIndex < currParticleLast && !(index < currParticleLast && index >= currParticleIndex)) ||
      (currParticleIndex > currParticleLast && (index >= currParticleLast && index < currParticleIndex))) {
    return;
  }

  float2 p = ((float2*)cuConstRendererParams.position)[index];
  int px = ceil(p.x);
  int py = ceil(p.y);

  float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (py * cuConstRendererParams.imageWidth + px)]);

  float4 tmpColor = ((float4*)cuConstRendererParams.colors)[index];

  *imgPtr = make_float4(tmpColor.x, tmpColor.y, tmpColor.z, tmpColor.w);
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

//render initialization function
SimRenderer::SimRenderer() {
  image = NULL;
  benchmark = STREAM1;

  numParticles = 0;

  cudaDeviceSpawners = NULL;
  cudaDevicePosition = NULL;
  cudaDeviceVelField = NULL;
  cudaDeviceColor = NULL;
  cudaDeviceImageData = NULL;
}

// render uninitialization func
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
    cudaFree(cudaDeviceSpawners);
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

    return image;
}

void
SimRenderer::loadScene(Benchmark bm, int maxArraySize) {
  loadParticleScene(bm, maxArraySize, image->width, image->height, numParticles, numSpawners, spawners, position, velField, color, isDynamic);
  maxNumParticles = maxArraySize;
  currParticleIndex = 0;
  currParticleLast = numParticles == 0 ? 1 : numParticles;
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

  cudaMalloc(&cudaDeviceSpawners, sizeof(float) * 2 * numSpawners);
  cudaMalloc(&cudaDevicePosition, sizeof(float) * 2 * maxNumParticles);
  cudaMalloc(&cudaDeviceVelField, sizeof(float) * 2 * image->width * image->height);
  cudaMalloc(&cudaDeviceColor, sizeof(float) * 4 * maxNumParticles);
  cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

  cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 2 * maxNumParticles, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceSpawners, spawners, sizeof(float) * 2 * numSpawners, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceVelField, velField, sizeof(float) * 2 * image->width * image->height, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 4 * maxNumParticles, cudaMemcpyHostToDevice);

  // Initialize parameters in constant memory.  We didn't talk about
  // constant memory in class, but the use of read-only constant
  // memory here is an optimization over just sticking these values
  // in device global memory.  NVIDIA GPUs have a few special tricks
  // for optimizing access to constant memory.  Using global memory
  // here would have worked just as well.  See the Programmer's
  // Guide for more information about constant memory.

  GlobalConstants params;
  params.benchmark = benchmark;
  params.numParticles = numParticles;
  params.numSpawners = numSpawners;
  params.imageWidth = image->width;
  params.imageHeight = image->height;
  params.spawners = cudaDeviceSpawners;
  params.position = cudaDevicePosition;
  params.velField = cudaDeviceVelField;
  params.colors = cudaDeviceColor;
  params.imageData = cudaDeviceImageData;
  params.maxNumParticles = maxNumParticles;

  cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
      printf("kernelCreateLinkedList launch failed with error \"%s\".\n",
             cudaGetErrorString(cudaerr));
}

void
SimRenderer::advanceAnimation() {
   // 256 threads per block is a healthy number
  dim3 blockDim(256, 1);
  int gridDimNum = maxNumParticles;
  dim3 gridDim((gridDimNum + blockDim.x - 1) / blockDim.x);

  kernelBasic<<<gridDim, blockDim>>>(currParticleIndex, currParticleLast);

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
  cudaError_t cudaerr;
  // 256 threads per block is a healthy number
  dim3 blockDimVec(8, 8);
  dim3 gridDimVec(64, 64);
  dim3 blockDimParticles(256);
  dim3 gridDimParticles(16, 16);

  float2* cudaDeviceVelFieldUpdated;

  cudaMalloc(&cudaDeviceVelFieldUpdated, sizeof(float) * 2 * image->width * image->height);

  //update momentum and copy the values accross whole frame
  kernelVecMomentum<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
  cudaDeviceSynchronize();
  kernelVelFieldCopy<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
  cudaDeviceSynchronize();

  //iteratively run divergence removal kernel
  for(int i = 0; i < ITERS; i++) {
    kernelUpdateVectorField<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
    cudaDeviceSynchronize();
    kernelVelFieldCopy<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
    cudaDeviceSynchronize();
  }

  cudaFree(cudaDeviceVelFieldUpdated);

  cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("vec field failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

  // if the kernel is dynamic (spawns particles), run the spawn particle function
  if(isDynamic && numSpawners != 0) {
    dim3 spawnGridDim(1);
    dim3 spawnBlockDim(numSpawners, 1);
    kernelSpawnParticles<<<spawnGridDim, spawnBlockDim>>>(currParticleIndex, currParticleLast);
    cudaDeviceSynchronize();

    int new_cpl = (currParticleLast + numSpawners - 1) % maxNumParticles;

    if(currParticleIndex < currParticleLast && (currParticleLast - currParticleIndex + numSpawners > maxNumParticles))
      currParticleLast = currParticleIndex;
    else if(currParticleIndex > currParticleLast && (currParticleIndex - currParticleLast >= numSpawners))
      currParticleLast = currParticleIndex;
    else
      currParticleLast = new_cpl;
  }

  //render the particles
  kernelRenderParticles<<<gridDimParticles, blockDimParticles>>>(currParticleIndex, currParticleLast);

  cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
}
