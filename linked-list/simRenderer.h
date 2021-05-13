struct Image;

typedef enum {
    STREAM1,
    STREAM2,
    STREAM4,
    CIRCLE,
    DYN1,
    DYN2,
    BLACKHOLE
} Benchmark;


class Node {
public:
    Node *prev;
    Node *next;

    float x;
    float y;

    float r;
    float g;
    float b;
    float a;
};

class List {
public:
    Node *head;
    Node *tail;
    int size;
};


class SimRenderer {
private:
    Image* image;
    Benchmark benchmark;

    int numParticles;
    int maxNumParticles;
    int numSpawners;

    int currParticleIndex;
    int currParticleLast;

    float* position;
    float* velField;
    float* color;
    float* spawners;

    float* cudaDevicePosition;
    float* cudaDeviceSpawners;
    float* cudaDeviceVelField;
    float* cudaDeviceColor;
    float* cudaDeviceImageData;

public:
    SimRenderer();
    virtual ~SimRenderer();

    const Image* getImage();

    void setup();

    void loadScene (Benchmark benchmark, int maxArraySize);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();
};
