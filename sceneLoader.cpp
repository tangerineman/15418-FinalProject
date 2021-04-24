#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <functional>

#
#include "simRenderer.h"


void loadParticleScene(
    Benchmark benchmark,
    int& numCircles,
    float*& position,
    float*& velocity,
    float*& color)
{
  printf("in loadParticleScene...\n");
  if (benchmark == SIMPLE) {
    numCircles = 512;

    position = new float[2*numCircles];
    velocity = new float[2*numCircles];
    color = new float[4*numCircles];

    for(int i = 0; i < numCircles; i++) {
      printf("loop iter: %d\n", i);
      int id2 = i * 2;
      int id4 = i * 4;
      position[id2] = (float) i; //x pos 100
      position[id2+1] = 64.f;
      velocity[id2] = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 700;
      velocity[id2+1] = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - .5f) * 700;
      color[id4] = 0.f;
      color[id4+1] = 0.f;
      color[id4+2] = 0.f;
      color[id4+3] = 1.f;
      if(i % 3 == 0) color[id4] = 1.f;
      if(i % 3 == 1) color[id4+1] = 1.f;
      if(i % 3 == 2) color[id4+2] = 1.f;
    }
  }
}
