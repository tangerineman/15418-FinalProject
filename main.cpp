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

  if (benchmarkStr.compare("stream1") == 0) {
    bm = STREAM1;
  } else if (benchmarkStr.compare("stream2") == 0) {
    bm = STREAM2;
  } else if (benchmarkStr.compare("circle") == 0) {
    bm = CIRCLE;
  } else {
    bm = STREAM1;
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
