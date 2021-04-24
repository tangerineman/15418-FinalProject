#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "simRenderer.h"
#include "platformgl.h"

void startRendererWithDisplay(SimRenderer* renderer);

int main(int argc, char** argv) {
  std::string benchmarkStr;

  printf("starting up...\n");

  Benchmark bm;
  int imageSize = 512;

  benchmarkStr = argv[1];

  SimRenderer* sim_renderer = new SimRenderer();

  if (benchmarkStr.compare("simple") == 0) {
    bm = SIMPLE;
  } else {
    bm = COMPLEX;
  }

  printf("allocating scene and stuff...\n");
  sim_renderer->allocOutputImage(imageSize, imageSize);
  printf("loading scene...\n");
  sim_renderer->loadScene(bm);
  printf("setting up...\n");
  sim_renderer->setup();

  printf("starting renderer...\n");

  glutInit(&argc, argv);
  startRendererWithDisplay(sim_renderer);

}
