#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <string>

#include "simRenderer.h"
#include "sceneLoader.h"

#include "image.h"

#define ITERS 20

struct GlobalConstants {
  Benchmark benchmark;

  float* velField;
  int numParticles;


  int imageWidth;
  int imageHeight;
  float* imageData;

  int* locks;
  float* initial_positions; // array of positions of particles that are pregenerated
  float* initial_colors;    // array of colors for particles that are pregenerated
};

__constant__ GlobalConstants cuConstRendererParams;

// linked list kernels
////////////////////////////////////////////////////////////////////////////////
__device__ List* particleList;

__device__ int deviceCurrNumParticles;
int currNumParticles;

// linked list initialization
// returns empty linked list with just a head (head contains no information)
__device__ List *linked_list_init(){

  List *newList = new(List);

  Node *newNode = new (Node);
  newNode->next = NULL;
  newNode->prev = NULL;
  newNode->x = 0;
  newNode->y = 0;
  newNode->r = 0;
  newNode->g = 0;
  newNode->b = 0;
  newNode->a = 0;

  newList->head = newNode;
  newList->tail = newNode;
  newList->size = 1;

  return newList;
}

// inserts a node at the tail of the linked list
__device__ void insert_node(List* list, float x, float y, float r, float g, float b, float a){

  Node *newNode = new (Node);

  newNode->prev = list->tail;
  newNode->next = NULL;
  newNode->x = x;
  newNode->y = y;

  newNode->r = r;
  newNode->g = g;
  newNode->b = b;
  newNode->a = a;

  list->tail->next = newNode;
  list->tail = newNode;
  list->size++;
  return;

}


// adds new particle with specified info to the linked list
__device__ void kernelAddParticle(float x, float y, float r, float g, float b, float a){

  if (blockIdx.x * blockDim.x + threadIdx.x == 0){
    insert_node(particleList, x, y, r, g, b, a);
    deviceCurrNumParticles++;
  }
}


// creates a linked list using the initial position and color arrays in global constants
__global__ void kernelCreateLinkedList() {

  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    deviceCurrNumParticles = 0;
    List *newList = linked_list_init();

    particleList = newList;

    float* positions = cuConstRendererParams.initial_positions;
    float* colors = cuConstRendererParams.initial_colors;

    for (int i = 0; i < cuConstRendererParams.numParticles; i++){

      int posIndex = 2 * i;
      int colIndex = 4 * i;

      float2 pos = make_float2(positions[posIndex], positions[posIndex + 1]);
      float4 color = make_float4(colors[colIndex], colors[colIndex+1],
                                  colors[colIndex+2], colors[colIndex+3]);

      kernelAddParticle(pos.x, pos.y, color.x, color.y, color.z, color.w);

    }

  }

  return;
}

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


// creates a particle at position x,y and adds it to the linked list
__global__ void kernelSpawnParticle(float x, float y){

  if (blockIdx.x * blockDim.x + threadIdx.x == 0){

    float *color = pickColor(particleList->size);
    kernelAddParticle(x, y, color[0], color[1], color[2], color[3]);
    free(color);

    }

  return;

}

__global__ void kernelFreeLinkedList() {
    if (blockIdx.x * blockDim.x + threadIdx.x == 0){
        Node *currNode = particleList->head;
        while (currNode != NULL){
            Node* next = currNode->next;
            delete currNode;
            currNode = next;
        }
        delete particleList;
    }
    return;
}

////////////////////////////////////////////////////////////////////////////////


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
//   float* position = cuConstRendererParams.position;

  int w = cuConstRendererParams.imageWidth;
  int h = cuConstRendererParams.imageHeight;

  Node *currNode = particleList->head->next;
  unsigned int thrId = blockIdx.x * blockDim.x + threadIdx.x;
  int index = 0;

  while (currNode != NULL){
    if (index % thrId == 0) {
      if ((index == 0 && thrId == 0) || (index != 0 && thrId != 0)){
        int currX = currNode->x;
        int currY = currNode->y;
        currNode->x += velField[2*(w * currY + currX)] * dt;

        if(ceil(currNode->x) <= 0)
          currNode->x = 0;
        else if (ceil(currNode->x) >= w-1)
          currNode->x = w-1;

        currNode->y += velField[2*(w * currY + currX) + 1] * dt;

        if(ceil(currNode->y) <= 0)
          currNode->y = 0;
        else if (ceil(currNode->y) >= h-1)
          currNode->y = h-1;
      }
    }
    currNode = currNode->next;
    index ++;
  }

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

//   int h = cuConstRendererParams.imageHeight;
  int w = cuConstRendererParams.imageWidth;

  int index = row * w + col;

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

__global__ void kernelRenderParticles() {
  Node *currNode = particleList->head->next;
  unsigned int thrId = blockIdx.x * blockDim.x + threadIdx.x;
  int index = 0;

  while (currNode != NULL){
    if (index % thrId == 0) {
      if ((index == 0 && thrId == 0) || (index != 0 && thrId != 0)){

        float2 p = make_float2(currNode->x, currNode->y);
        int px = ceil(p.x);
        int py = ceil(p.y);

        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (py * cuConstRendererParams.imageWidth + px)]);
        *imgPtr = make_float4(currNode->r, currNode->g, currNode->b, currNode->a);

      }
    }
    currNode = currNode->next;
    index ++;
  }

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
    kernelFreeLinkedList<<<1, 1>>>();
    // cudaFree(particleList);
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
SimRenderer::loadScene(Benchmark bm, int maxArraySize) {
    loadParticleScene(bm, maxArraySize, image->width, image->height, numParticles, numSpawners, spawners, position, velField, color, maxNumParticles);
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

  cudaMalloc(&cudaDevicePosition, sizeof(float) * 2 * numParticles);
  cudaMalloc(&cudaDeviceVelField, sizeof(float) * 2 * image->width * image->height);
  cudaMalloc(&cudaDeviceColor, sizeof(float) * 4 * numParticles);
  cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

  cudaMemcpy(cudaDeviceVelField, velField, sizeof(float) * 2 * image->width * image->height, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 4 * numParticles, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 2 * numParticles, cudaMemcpyHostToDevice);


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
  params.imageWidth = image->width;
  params.imageHeight = image->height;
  params.initial_positions = cudaDevicePosition;
  params.velField = cudaDeviceVelField;
  params.initial_colors = cudaDeviceColor;
  params.imageData = cudaDeviceImageData;

  cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));
  currNumParticles = 0;
  kernelCreateLinkedList<<<1, 1>>>();
  cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernelCreateLinkedList launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

}

void
SimRenderer::advanceAnimation() {
   // 256 threads per block is a healthy number
  dim3 blockDim(256, 1);
  int gridDimNum = currNumParticles > numParticles ? currNumParticles: numParticles;
  dim3 gridDim((gridDimNum + blockDim.x - 1) / blockDim.x);


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
  cudaError_t cudaerr;
  printf("IN RENDERER\n");

  dim3 blockDimVec(8, 8);
  dim3 gridDimVec(64, 64);
  dim3 blockDimParticles(256);
  dim3 gridDimParticles(16, 16);

  // spawn particles based on spawner locations
  for (int j = 0; j < numSpawners; j++){
      if (currNumParticles > maxNumParticles) break;
      float newX = spawners[2 * j];
      float newY = spawners[2 * j + 1];
      kernelSpawnParticle<<<1, 1>>>(newX, newY);
      currNumParticles++;
      cudaerr = cudaDeviceSynchronize();
      if (cudaerr != cudaSuccess)
          printf("kernelSpawnParticle launch failed with error \"%s\".\n",
              cudaGetErrorString(cudaerr));
  }





  float2* cudaDeviceVelFieldUpdated;

  cudaMalloc(&cudaDeviceVelFieldUpdated, sizeof(float) * 2 * image->width * image->height);

  kernelVecMomentum<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
  cudaDeviceSynchronize();
  kernelVelFieldCopy<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
  cudaDeviceSynchronize();

  for(int i = 0; i < ITERS; i++) {
    kernelUpdateVectorField<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
    cudaDeviceSynchronize();
    kernelVelFieldCopy<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
    cudaDeviceSynchronize();
  }

  cudaFree(cudaDeviceVelFieldUpdated);

  cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

  kernelRenderParticles<<<gridDimParticles, blockDimParticles>>>();
  cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
}
