#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "simRenderer.h"

void startRendererWithDisplay(SimRenderer* renderer);

int main(int argc, char** argv) {
  std::string benchmarkStr;

  Benchmark bm;
  int imageSize = 512;

  benchmarkStr = argv[1];

  SimRenderer* sim_renderer = new SimRenderer();

  if (benchmarkStr.compare("simple") == 0) {
    bm = SIMPLE;
  } else {
    bm = COMPLEX;
  }

  sim_renderer->allocOutputImage(imageSize, imageSize);
  sim_renderer->loadScene(bm);
  sim_renderer->setup();

  startRendererWithDisplay(sim_renderer);

}
