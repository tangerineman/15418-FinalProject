#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <functional>

#include "simRenderer.h"


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
  if (benchmark == SIMPLE) {
    numParticles = 512;

    position = new float[2*numParticles];
    velField = new float[2*width*height];
    color = new float[4*numParticles];

    for(int i = 0; i < numParticles; i++) {
      printf("loop iter: %d\n", i);
      int id2 = i * 2;
      int id4 = i * 4;

      if(i % 4 == 0) {
        position[id2] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200;
        position[id2+1] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200;
        color[id4] = 1.f;
        color[id4+1] = 0.f;
        color[id4+2] = 0.f;
        color[id4+3] = 1.f;
      }
      if(i % 4 == 1) {
        position[id2] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200 + (511 - 200);
        position[id2+1] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 200;
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
      /*position[id2] = (float) i; //x pos 100
      //position[id2+1] = 256.f;
      //velocity[id2] = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 700;
      //velocity[id2+1] = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 700;
      color[id4] = 0.f;
      color[id4+1] = 0.f;
      color[id4+2] = 0.f;
      color[id4+3] = 1.f;
      if(i % 3 == 0) color[id4] = 1.f;
      if(i % 3 == 1) color[id4+1] = 1.f;
      if(i % 3 == 2) color[id4+2] = 1.f;*/
    }

    for(int j = 0; j < height; j++) {
      for(int k = 0; k < width; k++) {
        if (j < height/2 && k < width/2) {
          velField[(j*width + k) * 2] = -250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
          velField[(j*width + k) * 2 + 1] = 250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
        } else if (j > height/2 && k < width/2) {
          velField[(j*width + k) * 2] = 250.f;//  + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
          velField[(j*width + k) * 2 + 1] = 250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
        } else if (j < height/2 && k > width/2) {
          velField[(j*width + k) * 2] = -250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
          velField[(j*width + k) * 2 + 1] = -250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
        } else if (j > height/2 &&  k > width/2) {
          velField[(j*width + k) * 2] = 250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
          velField[(j*width + k) * 2 + 1] = -250.f;// + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 100;
        } else {
          velField[(j*width + k) * 2] = 0.f;
          velField[(j*width + k) * 2 + 1] = 0.f;
        }
        /*
        if (j < height/2 - 5) {
          velField[(j*width + k) * 2] = 0.f;//((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 300;
          velField[(j*width + k) * 2 + 1] = 0.f;//((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 300;
        } else if (j > height/2+5) {
          velField[(j*width + k) * 2] = 0.f;
          velField[(j*width + k) * 2 + 1] = 0.f;
        } else if (k < width/2){
          velField[(j*width + k) * 2] = 500.f;
          velField[(j*width + k) * 2 + 1] = 0.f;
        } else if (k > width/2) {
          velField[(j*width + k) * 2] = -500.f;
          velField[(j*width + k) * 2 + 1] = 0.f;
        }
        //velField[j*2] = 0;//((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 300;
        //velField[j*2+1] = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 300;
        */
      }
    }
  }
}
