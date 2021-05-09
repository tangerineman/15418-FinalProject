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
    float*& color)
{
  printf("in loadParticleScene...\n");
  if (benchmark == STREAM1) {
    numParticles = 256;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    int stream_height = 8;
    int stream_start_y = (height+stream_height)/2;
    int stream_width = 200;

    for(int i = 0; i < numParticles; i++) {
      int id2 = i * 2;
      int id4 = i * 4;

      position[id2] = genRandFloat(0, stream_width);
      printf("posx: %f\n", position[id2]);
      position[id2+1] = genRandFloat(stream_start_y, stream_start_y+stream_height);

      color[id4] = 0.f;
      color[id4+1] = 1.f;
      color[id4+2] = 0.f;
      color[id4+3] = 1.f;
    }

    for(int j = 0; j < height; j++) {
      for(int k = 0; k < width; k++) {
        if(j >= stream_start_y && j <= (stream_start_y + stream_height) && k < stream_width) {
          velField[(j*width + k) * 2] = 100.f;
          velField[((j*width + k) * 2) + 1] = 0.f;
        } else {
          velField[(j*width + k) * 2] = 0.f;
          velField[((j*width + k) * 2) + 1] = 0.f;
        }
      }
    }

  } else if (benchmark == STREAM2) {
    numParticles = 256;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    int stream_start_y = 255;
    int stream_height = 5;
    int stream_width = 250;

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
            velField[(j*width + k) * 2] = 100.f;
          else if (k > (width - stream_width))
            velField[(j*width + k) * 2] = -100.f;
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

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    for(int i = 0; i < numParticles; i++) {
      int id2 = i * 2;
      int id4 = i * 4;

      if(i % 4 == 0) {
        position[id2] = genRandFloat(0, 200);
        position[id2+1] = genRandFloat(0, 200);
        color[id4] = 1.f;
        color[id4+1] = 0.f;
        color[id4+2] = 0.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 1) {
        position[id2] = genRandFloat(width-200, width);//(static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200 + (511 - 200);
        position[id2+1] = genRandFloat(0, 200);//(static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200;
        color[id4] = 0.f;
        color[id4+1] = 1.f;
        color[id4+2] = 0.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 2) {
        position[id2] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200;
        position[id2+1] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200 + (511 - 200);
        color[id4] = 0.f;
        color[id4+1] = 0.f;
        color[id4+2] = 1.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 3) {
        position[id2] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200 + (511 - 200);
        position[id2+1] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200 + (511 - 200);
        color[id4] = 0.f;
        color[id4+1] = .5;
        color[id4+2] = .5;
        color[id4+3] = 1.f;
      }
    }

    for(int j = 0; j < height; j++) {
      for(int k = 0; k < width; k++) {
        if (j < height/2 && k <= width/2) {
          velField[(j*width + k) * 2] = 250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
          velField[(j*width + k) * 2 + 1] = 0.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
        } else if (j > height/2 && k <= width/2) {
          velField[(j*width + k) * 2] = 0.f;//  + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
          velField[(j*width + k) * 2 + 1] = -250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
        } else if (j <= height/2 && k > width/2) {
          velField[(j*width + k) * 2] = 0.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
          velField[(j*width + k) * 2 + 1] = 250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
        } else if (j >= height/2 &&  k > width/2) {
          velField[(j*width + k) * 2] = -250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
          velField[(j*width + k) * 2 + 1] = 0.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
        } else {
          velField[(j*width + k) * 2] = 0.f;
          velField[(j*width + k) * 2 + 1] = 0.f;
        }
      }
    }
  }
}
