#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <functional>

#include "simRenderer.h"

float genRandFloat(int low, int high) {
  return low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(high-low)));
}

void loadParticleScene(
    Benchmark benchmark,
    int width,
    int height,
    int& numParticles,
    float*& position,
    float*& velField,
    float*& color,
    bool& isDynamic,
    int& numSpawners,
    float*& spawners)
{
  printf("in loadParticleScene...\n");
  if (benchmark == STREAM1) {
    numParticles = 256;
    numSpawners = 0;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    isDynamic = false;

    int stream_height = 5;
    int stream_start_y = (height-stream_height)/2;
    int stream_width = 200;

    for(int i = 0; i < numParticles; i++) {
      int id2 = i * 2;
      int id4 = i * 4;

      position[id2] = genRandFloat(0, stream_width);
      //printf("posx: %f\n", position[id2]);
      position[id2+1] = genRandFloat(stream_start_y, stream_start_y+stream_height);

      color[id4] = 0.f;
      color[id4+1] = 1.f;
      color[id4+2] = 0.f;
      color[id4+3] = 1.f;
    }

    for(int j = 0; j < height; j++) {
      for(int k = 0; k < width; k++) {
        if(j >= stream_start_y && j <= (stream_start_y + stream_height) && k < stream_width) {
          velField[(j*width + k) * 2] = 160.f;
          velField[((j*width + k) * 2) + 1] = 0.f;
        } else {
          velField[(j*width + k) * 2] = 0.f;
          velField[((j*width + k) * 2) + 1] = 0.f;
        }
      }
    }

  } else if (benchmark == STREAM2) {
    numParticles = 2048;
    numSpawners = 0;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    isDynamic = false;

    int stream_height = 5;
    int stream_start_y = (height-stream_height)/2;
    int stream_width = 230;

    for(int i = 0; i < numParticles; i++) {
      int id2 = i * 2;
      int id4 = i * 4;

      color[id4] = 0.f;
      color[id4+1] = 0.f;
      color[id4+2] = 0.f;
      color[id4+3] = 1.f;

      if(i % 2) {
        position[id2] = genRandFloat(0, stream_width);
        color[id4+1] = 1.f;
      } else {
        position[id2] = genRandFloat(width-stream_width, width);
        color[id4+2] = 1.f;
      }

      position[id2+1] = genRandFloat(stream_start_y, stream_start_y+stream_height);


    }

    for(int j = 0; j < height; j++) {
      for(int k = 0; k < width; k++) {
        if(j >= stream_start_y && j <= (stream_start_y + stream_height)) {
          if (k < stream_width)
            velField[(j*width + k) * 2] = 160.f;
          else if (k > (width - stream_width))
            velField[(j*width + k) * 2] = -160.f;
          else
            velField[(j*width + k) * 2] = 0.f;

          velField[((j*width + k) * 2) + 1] = 0.f;
        } else {
          velField[(j*width + k) * 2] = 0.f;
          velField[((j*width + k) * 2) + 1] = 0.f;
        }
      }
    }
  } else if (benchmark == CIRCLE) {
    numParticles = 1024;
    numSpawners = 0;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];
    
    isDynamic = false;

    for(int i = 0; i < numParticles; i++) {
      int id2 = i * 2;
      int id4 = i * 4;

      if(i % 4 == 0) {
        position[id2] = genRandFloat(0, width/2);
        position[id2+1] = genRandFloat(0, height/2);
        color[id4] = 1.f;
        color[id4+1] = 0.f;
        color[id4+2] = 0.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 1) {
        position[id2] = genRandFloat(width/2, width);
        position[id2+1] = genRandFloat(0, height/2);
        color[id4] = 0.f;
        color[id4+1] = 1.f;
        color[id4+2] = 0.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 2) {
        position[id2] = genRandFloat(0, width/2);
        position[id2+1] = genRandFloat(height/2, height);
        color[id4] = 0.f;
        color[id4+1] = 0.f;
        color[id4+2] = 1.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 3) {
        position[id2] = genRandFloat(width/2, width);
        position[id2+1] = genRandFloat(height/2, height);
        color[id4] = 0.f;
        color[id4+1] = .5;
        color[id4+2] = .5;
        color[id4+3] = 1.f;
      }
    }

    float centerX = (float)(width / 2);
    float centerY = (float)(height / 2);
    float vecScale = -5000.f;
    // initialize radial vector field
    for(int j = 0; j < height; j++) {
      for(int k = 0; k < width; k++) {

        float offsetX = (float)k - centerX;
        float offsetY = (float)j - centerY;

        velField[(j*width + k) * 2] = offsetY / (offsetX * offsetX + offsetY * offsetY)  * vecScale;
        velField[(j*width + k) * 2 + 1] = (-1.f * offsetX) / (offsetX * offsetX + offsetY * offsetY)  * vecScale;
      }
    }


  } else if (benchmark == BLACKHOLE){
    numParticles = 1024;
    numSpawners = 0;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];
    
    isDynamic = false;

    for(int i = 0; i < numParticles; i++) {
      int id2 = i * 2;
      int id4 = i * 4;

      if(i % 4 == 0) {
        position[id2] = genRandFloat(0, width/2);
        position[id2+1] = genRandFloat(0, height/2);
        color[id4] = 1.f;
        color[id4+1] = 0.f;
        color[id4+2] = 0.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 1) {
        position[id2] = genRandFloat(width/2, width);
        position[id2+1] = genRandFloat(0, height/2);
        color[id4] = 0.f;
        color[id4+1] = 1.f;
        color[id4+2] = 0.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 2) {
        position[id2] = genRandFloat(0, width/2);
        position[id2+1] = genRandFloat(height/2, height);
        color[id4] = 0.f;
        color[id4+1] = 0.f;
        color[id4+2] = 1.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 3) {
        position[id2] = genRandFloat(width/2, width);
        position[id2+1] = genRandFloat(height/2, height);
        color[id4] = 0.f;
        color[id4+1] = .5;
        color[id4+2] = .5;
        color[id4+3] = 1.f;
      }
    }

    float centerX = (float)(width / 2);
    float centerY = (float)(height / 2);
    // float vecScale = -5000.f;
    float vecScale = -1.f;
    // initialize radial vector field
    for(int j = 0; j < height; j++) {
      for(int k = 0; k < width; k++) {

        float offsetX = (float)k - centerX;
        float offsetY = (float)j - centerY;

        velField[(j*width + k) * 2] = offsetX * vecScale;
        velField[(j*width + k) * 2 + 1] = vecScale * (offsetY - offsetX);
      }
    }
  } else if (benchmark == DYN1){

      numParticles = 0;
      
      color = new float;
      velField = new float[2*width*height];

      numSpawners = 8;
      position = new float[2 * numSpawners];
      isDynamic = true;

      float centerOffset = 100.f;
      float centerX = (float)(width / 2);
      float centerY = (float)(height / 2);

      // place spawners
      position[0] = centerX + centerOffset;
      position[1] = centerY + centerOffset;

      position[2] = centerX - centerOffset;
      position[3] = centerY - centerOffset;

      position[4] = centerX - centerOffset;
      position[5] = centerY + centerOffset;

      position[6] = centerX + centerOffset;
      position[7] = centerY - centerOffset;

      position[8] = centerX;
      position[9] = centerY + centerOffset;

      position[10] = centerX;
      position[11] = centerY - centerOffset;

      position[12] = centerX - centerOffset;
      position[13] = centerY;

      position[14] = centerX + centerOffset;
      position[15] = centerY;


      float vecScale = 0.1;
      for(int j = 0; j < height; j++) {
        for(int k = 0; k < width; k++) {

          float offsetX = (float)k - centerX;
          float offsetY = (float)j - centerY;

          velField[(j*width + k) * 2] = vecScale * (offsetX * offsetX + offsetY * offsetY);
          velField[(j*width + k) * 2 + 1] = vecScale * offsetX * offsetY;
        }
      }
      
  }


  else if (benchmark == DYN2){

      numParticles = 0;
      
      color = new float;
      velField = new float[2*width*height];

      numSpawners = 8;
      position = new float[2 * numSpawners];
      isDynamic = true;

      float centerOffset = 50.f;
      float centerX = (float)(width / 2);
      float centerY = (float)(height / 2);

      // place spawners
      position[0] = centerX + centerOffset;
      position[1] = centerY + centerOffset;

      position[2] = centerX - centerOffset;
      position[3] = centerY - centerOffset;

      position[4] = centerX - centerOffset;
      position[5] = centerY + centerOffset;

      position[6] = centerX + centerOffset;
      position[7] = centerY - centerOffset;

      position[8] = centerX;
      position[9] = centerY + centerOffset;

      position[10] = centerX;
      position[11] = centerY - centerOffset;

      position[12] = centerX - centerOffset;
      position[13] = centerY;

      position[14] = centerX + centerOffset;
      position[15] = centerY;


      float vecScale = 5000.f;
      // initialize radila vector field
      for(int j = 0; j < height; j++) {
        for(int k = 0; k < width; k++) {

          float offsetX = (float)k - centerX;
          float offsetY = (float)j - centerY;
          velField[(j*width + k) * 2] = offsetY / (offsetX * offsetX + offsetY * offsetY)  * vecScale;
          velField[(j*width + k) * 2 + 1] = (-1.f * offsetX) / (offsetX * offsetX + offsetY * offsetY)  * vecScale;
        }
      }
      
      
  }
}
