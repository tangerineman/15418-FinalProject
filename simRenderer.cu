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
  int initNumParticles;

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

__device__ int currNumParticles;

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

//   if (list == NULL) printf("ERROR: NULL LIST IN INSERT_NODE\n");
//   else if (list->tail == NULL) printf("ERROR: NULL LIST->NEXT IN INSERT_NODE\n");
  list->tail->next = newNode;
  list->tail = newNode;
  list->size++;
  // printf("size = %d", list->size);
  return;

}


__device__ void kernelAddParticle(float x, float y, float r, float g, float b, float a){
  
  if (blockIdx.x * blockDim.x + threadIdx.x == 0){
    insert_node(particleList, x, y, r, g, b, a);
    currNumParticles++;
  }
}


// creates a linked list using the initial position and color arrays in global constants
__global__ void kernelCreateLinkedList() {
 
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    // printf("top of if\n");
    currNumParticles = 0;
    List *newList = linked_list_init();

    // printf("Newlist = %p\n", (void*)newList);
    // printf("cuConstRendererParams.particleList = %p\n", particleList);
    particleList = newList;
    
    // printf("after init\n");
    // printf("THREAD: %d\n", blockIdx.x * blockDim.x + threadIdx.x);
    float* positions = cuConstRendererParams.initial_positions;
    float* colors = cuConstRendererParams.initial_colors;

    for (int i = 0; i < cuConstRendererParams.initNumParticles; i++){

      int posIndex = 2 * i;
      int colIndex = 4 * i;

      float2 pos = make_float2(positions[posIndex], positions[posIndex + 1]);
      float4 color = make_float4(colors[colIndex], colors[colIndex+1],
                                  colors[colIndex+2], colors[colIndex+3]);
                  
        
      kernelAddParticle(pos.x, pos.y, color.x, color.y, color.z, color.w);
      // printf("size = %d\n", cuConstRendererParams.particleList->size);
    }

  }
//   printf("done\n");
  return;
}



__global__ void freeLinkedList() {

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

  while (index < particleList->size && currNode != NULL){
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
  // printf("KERNEL RENDER PARTICLES\n");

  Node *currNode = particleList->head->next;
  unsigned int thrId = blockIdx.x * blockDim.x + threadIdx.x;
  int index = 0;

  while (index < particleList->size && currNode != NULL){
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

  initNumParticles = 0;

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
  loadParticleScene(bm, image->width, image->height, initNumParticles, position, velField, color);
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

  cudaMalloc(&cudaDevicePosition, sizeof(float) * 2 * initNumParticles);
  cudaMalloc(&cudaDeviceVelField, sizeof(float) * 2 * image->width * image->height);
  cudaMalloc(&cudaDeviceColor, sizeof(float) * 4 * initNumParticles);
  cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

  cudaMemcpy(cudaDeviceVelField, velField, sizeof(float) * 2 * image->width * image->height, cudaMemcpyHostToDevice);
  
  cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 4 * initNumParticles, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 2 * initNumParticles, cudaMemcpyHostToDevice);

  cudaMalloc(&particleList, sizeof(List*));
  
  
  // Initialize parameters in constant memory.  We didn't talk about
  // constant memory in class, but the use of read-only constant
  // memory here is an optimization over just sticking these values
  // in device global memory.  NVIDIA GPUs have a few special tricks
  // for optimizing access to constant memory.  Using global memory
  // here would have worked just as well.  See the Programmer's
  // Guide for more information about constant memory.

  GlobalConstants params;
  params.benchmark = benchmark;
  params.initNumParticles = initNumParticles;
  params.imageWidth = image->width;
  params.imageHeight = image->height;
  params.initial_positions = cudaDevicePosition;
  params.velField = cudaDeviceVelField;
  params.initial_colors = cudaDeviceColor;
  params.imageData = cudaDeviceImageData;

  cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

  kernelCreateLinkedList<<<1, 1>>>();

  cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernelCreateLinkedList launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

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
  dim3 gridDim((initNumParticles + blockDim.x - 1) / blockDim.x);

  


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
  cudaError_t cudaerr;
  cudaMalloc(&cudaDeviceVelFieldUpdated, sizeof(float) * 2 * image->width * image->height);

  for(int i = 0; i < 30; i++) {
    kernelUpdateVectorField<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernelUpdateVectorField launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    kernelVelFieldCopy<<<gridDimVec, blockDimVec>>>(cudaDeviceVelFieldUpdated);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernelVelFieldCopy launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    //cudaMemcpy(cuConstRendererParams.velField, cudaDeviceVelFieldUpdated, sizeof(float) * 2 * image->width * image->height, cudaMemcpyDeviceToDevice);
  }

  cudaFree(cudaDeviceVelFieldUpdated);

  cudaerr = cudaDeviceSynchronize();
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
