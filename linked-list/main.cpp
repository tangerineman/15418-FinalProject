#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "simRenderer.h"
#include "platformgl.h"

void startRendererWithDisplay(SimRenderer* renderer);

int main(int argc, char** argv) {
  std::string benchmarkStr;
  int arrayMaxSize;

  printf("starting up...\n");

  Benchmark bm;
  int imageSize = 512;

  if (argc > 1) benchmarkStr = argv[1];
  if (argc > 2) arrayMaxSize = atoi(argv[2]);
  else arrayMaxSize = 1024;
  SimRenderer* sim_renderer = new SimRenderer();


  if (argc > 1){
    printf("Running benchmark: ");
    if (benchmarkStr.compare("stream1") == 0) {
    bm = STREAM1;
    printf("stream1\n");
  } else if (benchmarkStr.compare("stream2") == 0) {
    bm = STREAM2;
    printf("stream2\n");
  } else if (benchmarkStr.compare("stream4") == 0) {
    bm = STREAM4;
    printf("stream4\n");
  } else if (benchmarkStr.compare("circle") == 0) {
    bm = CIRCLE;
    printf("circle\n");
  } else if (benchmarkStr.compare("blackhole") == 0) {
    bm = BLACKHOLE;
    printf("circle\n");
  } else if (benchmarkStr.compare("dyn1") == 0) {
    bm = DYN1;
    printf("dyn1\n");
  } else if (benchmarkStr.compare("dyn2") == 0) {
    bm = DYN2;
    printf("dyn2\n");
  } else {
    bm = STREAM1;
    printf("stream1\n");
  }
  }
  else {
      bm = STREAM1;
      printf("Running benchmark: stream1\n");
  }



  printf("allocating scene and stuff...\n");
  sim_renderer->allocOutputImage(imageSize, imageSize);
  printf("loading scene...\n");
  sim_renderer->loadScene(bm, arrayMaxSize);
  printf("setting up...\n");
  sim_renderer->setup();
  

  printf("starting renderer...\n");

  glutInit(&argc, argv);
  startRendererWithDisplay(sim_renderer);

}
