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
  if (benchmark == SIMPLE) {
    numCircles = 256;

    for(int i = 0; i < numCircles; i++) {
      int id2 = i * 2;
      int id4 = i * 4;
      position[id2] = (float) (i+10); //x pos 100
      position[id2+1] = 64.f;
      velocity[id2] = 0.f;
      velocity[id2+1] = 0.f;
      if(i % 3 == 0) color[id4] = 1.f;
      if(i % 3 == 1) color[id4+1] = 1.f;
      if(i % 3 == 2) color[id4+2] = 1.f;
    }
  }
}
