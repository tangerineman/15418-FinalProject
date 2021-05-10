#ifndef __SCENE_LOADER_H__
#define __SCENE_LOADER_H__


void
loadParticleScene(
    Benchmark benchmark,
    int width,
    int height,
    int& numCircles,
    float*& position,
    float*& velocity,
    float*& color,
    bool& isDynamic,
    int & numSpawners);

#endif
