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
    int maxArraySize,
    int width,
    int height,
    int& numParticles,
    int& numSpawners,
    float*& spawners,
    float*& position,
    float*& velField,
    float*& color,
    int& maxNumParticles
    )
{
  printf("in loadParticleScene...\n");
  if (benchmark == STREAM1) {
    numParticles = 1024;
    numSpawners = 0;
    
    spawners = new float[2*numSpawners];
    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    maxNumParticles = maxArraySize;

    int stream_height = 5;
    int stream_start_y = (height-stream_height)/2;
    int stream_width = 245;
    
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
          velField[(j*width + k) * 2] = genRandFloat(255, 265);//160.f;
          velField[((j*width + k) * 2) + 1] = genRandFloat(-5, 5);
        } else {
          velField[(j*width + k) * 2] = genRandFloat(-5, 5);
          velField[((j*width + k) * 2) + 1] = genRandFloat(-5, 5);
        }
      }
    }

  } else if (benchmark == STREAM2) {
    numParticles = 2048;
    numSpawners = 0;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    maxNumParticles = maxArraySize;

    int stream_height = 5;
    int stream_start_y = (height-stream_height)/2;
    int stream_width = 245;

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
            velField[(j*width + k) * 2] = genRandFloat(250, 260);//160.f;
          else if (k > (width - stream_width))
            velField[(j*width + k) * 2] = genRandFloat(-250, -260);//160.f;
          else
            velField[(j*width + k) * 2] = genRandFloat(-5, 5);

          velField[((j*width + k) * 2) + 1] = genRandFloat(-5, 5);
        } else {
          velField[(j*width + k) * 2] = genRandFloat(-5, 5);;
          velField[((j*width + k) * 2) + 1] = genRandFloat(-5, 5);
        }
      }
    }
  } else if (benchmark == STREAM4) {
    numParticles = 4096;
    numSpawners = 0;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    maxNumParticles = maxArraySize;

    int stream_height = 5;
    int stream_start_y = (height-stream_height)/2;
    int stream_width = 235;

    for(int i = 0; i < numParticles; i++) {
      int id2 = i * 2;
      int id4 = i * 4;

      color[id4] = 0.f;
      color[id4+1] = 0.f;
      color[id4+2] = 0.f;
      color[id4+3] = 1.f;

      if(i % 4 == 0) {
        position[id2] = genRandFloat(0, stream_width);
        position[id2+1] = genRandFloat(stream_start_y, stream_start_y+stream_height);
        color[id4+1] = 1.f;
      } else if(i%4 == 1){
        position[id2] = genRandFloat(width-stream_width, width);
        position[id2+1] = genRandFloat(stream_start_y, stream_start_y+stream_height);
        color[id4+2] = 1.f;
      } else if(i%4 == 2) {
        position[id2] = genRandFloat(stream_start_y, stream_start_y+stream_height);
        position[id2+1] = genRandFloat(0, width);
        color[id4] = 1.f;
      } else {
        position[id2] = genRandFloat(stream_start_y, stream_start_y+stream_height);
        position[id2+1] = genRandFloat(width-stream_width, width);
        color[id4] = 1.f;
        color[id4+2] = 1.f;
      }


    }

    for(int j = 0; j < height; j++) {
      for(int k = 0; k < width; k++) {
        if(j >= stream_start_y && j <= (stream_start_y + stream_height)) {
          if (k < stream_width)
            velField[(j*width + k) * 2] = genRandFloat(250, 260);//160.f;
          else if (k > (width - stream_width))
            velField[(j*width + k) * 2] = genRandFloat(-250, -260);//160.f;
          else
            velField[(j*width + k) * 2] = genRandFloat(-5, 5);

          velField[((j*width + k) * 2) + 1] = genRandFloat(-5, 5);
        } else if(k >= stream_start_y && k <= (stream_start_y + stream_height)) {
          if (j < stream_width)
            velField[(j*width + k) * 2 + 1] = genRandFloat(250, 260);//160.f;
          else if (j > (width - stream_width))
            velField[(j*width + k) * 2 + 1] = genRandFloat(-250, -260);//160.f;
          else
            velField[(j*width + k) * 2 + 1] = genRandFloat(-5, 5);

          velField[((j*width + k) * 2)] = genRandFloat(-5, 5);
        } else {
          velField[(j*width + k) * 2] = genRandFloat(-5, 5);;
          velField[((j*width + k) * 2) + 1] = genRandFloat(-5, 5);
        }
      }
    }
  } else if (benchmark == CIRCLE) {
    numParticles = 2048;
    numSpawners = 0;

    position = new float[2*maxArraySize];
    velField = new float[2*width*height];
    color = new float[4*maxArraySize];

    maxNumParticles = maxArraySize;

    for(int i = 0; i < maxArraySize; i++) {
      int id2 = i * 2;
      int id4 = i * 4;

      if(i % 4 == 0) {
        position[id2] = genRandFloat(0, width/2);
        position[id2+1] = genRandFloat(0, height/2);
        color[id4] = .5f;
        color[id4+1] = 0.f;
        color[id4+2] = 0.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 1) {
        position[id2] = genRandFloat(width/2, width);
        position[id2+1] = genRandFloat(0, height/2);
        color[id4] = 0.f;
        color[id4+1] = .5f;
        color[id4+2] = 0.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 2) {
        position[id2] = genRandFloat(0, width/2);
        position[id2+1] = genRandFloat(height/2, height);
        color[id4] = 0.f;
        color[id4+1] = 0.f;
        color[id4+2] = .5f;
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
        float v1, v2;
        v1 = offsetY / (offsetX * offsetX + offsetY * offsetY)  * vecScale;
        v2 = (-1.f * offsetX) / (offsetX * offsetX + offsetY * offsetY)  * vecScale;

        v1 = (v1 > 300.f || isnan(v1)) ? 300.f : (v1 < -300.f ? -300.f : v1);
        v2 = (v2 > 300.f || isnan(v2)) ? 300.f : (v2 < -300.f ? -300.f : v2);

        velField[(j*width + k) * 2] = v1;
        velField[(j*width + k) * 2 + 1] =  v2;
      }
    }


  } else if (benchmark == BLACKHOLE){
    numParticles = 8192;
    numSpawners = 0;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    maxNumParticles = maxArraySize;

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

      color = new float[4*numParticles];
      velField = new float[2*width*height];

      numSpawners = 8;
      position = new float[2 * numParticles];
      spawners = new float[2 * numSpawners];
      maxNumParticles = maxArraySize;

      float centerOffset = 100.f;
      float centerX = (float)(width / 2);
      float centerY = (float)(height / 2);

      // place spawners
      spawners[0] = centerX + centerOffset;
      spawners[1] = centerY + centerOffset;

      spawners[2] = centerX - centerOffset;
      spawners[3] = centerY - centerOffset;

      spawners[4] = centerX - centerOffset;
      spawners[5] = centerY + centerOffset;

      spawners[6] = centerX + centerOffset;
      spawners[7] = centerY - centerOffset;

      spawners[8] = centerX;
      spawners[9] = centerY + centerOffset;

      spawners[10] = centerX;
      spawners[11] = centerY - centerOffset;

      spawners[12] = centerX - centerOffset;
      spawners[13] = centerY;

      spawners[14] = centerX + centerOffset;
      spawners[15] = centerY;


      float vecScale = -1.f;
      for(int j = 0; j < height; j++) {
        for(int k = 0; k < width; k++) {

          float offsetX = (float)k - centerX;
          float offsetY = (float)j - centerY;

          velField[(j*width + k) * 2] = vecScale * offsetX;
          velField[(j*width + k) * 2 + 1] = vecScale * offsetY;
        }
      }

  }


  else if (benchmark == DYN2){

      numParticles = 0;

      color = new float[4*numParticles];
      velField = new float[2*width*height];

      numSpawners = 8;
      position = new float[2 * numParticles];
      spawners = new float[2 * numSpawners];
      maxNumParticles = maxArraySize;

      float centerOffset = 50.f;
      float centerX = (float)(width / 2);
      float centerY = (float)(height / 2);

      // place spawners
      spawners[0] = centerX + centerOffset;
      spawners[1] = centerY + centerOffset;

      spawners[2] = centerX - centerOffset;
      spawners[3] = centerY - centerOffset;

      spawners[4] = centerX - centerOffset;
      spawners[5] = centerY + centerOffset;

      spawners[6] = centerX + centerOffset;
      spawners[7] = centerY - centerOffset;

      spawners[8] = centerX;
      spawners[9] = centerY + centerOffset;

      spawners[10] = centerX;
      spawners[11] = centerY - centerOffset;

      spawners[12] = centerX - centerOffset;
      spawners[13] = centerY;

      spawners[14] = centerX + centerOffset;
      spawners[15] = centerY;


      float vecScale = 5000.f;
      printf("setup w, h: %d, %d\n", width, height);
      // initialize radila vector field
      for(int j = 0; j < height; j++) {
        for(int k = 0; k < width; k++) {

          float offsetX = (float)k - centerX;
          float offsetY = (float)j - centerY;

          float v1, v2;
          v1 = offsetY / (offsetX * offsetX + offsetY * offsetY)  * vecScale;
          v2 = (-1.f * offsetX) / (offsetX * offsetX + offsetY * offsetY)  * vecScale;

          v1 = (v1 > 300.f || isnan(v1)) ? 300.f : (v1 < -300.f ? -300.f : v1);
          v2 = (v2 > 300.f || isnan(v2)) ? 300.f : (v2 < -300.f ? -300.f : v2);

          velField[(j*width + k) * 2] = v1;
          velField[(j*width + k) * 2 + 1] = v2;
        }
      }


  }

  printf("DONE LOADING SCENE\n");
}
