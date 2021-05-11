#ifndef __SCENE_LOADER_H__
#define __SCENE_LOADER_H__


void
loadParticleScene(
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
  bool& isDynamic);

#endif
