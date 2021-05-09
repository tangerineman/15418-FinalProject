struct Image;

typedef enum {
    STREAM1,
    STREAM2,
    CIRCLE,
    DYN1
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

    int initNumParticles;

    bool *isDynamic;

    float* position;
    float* velField;
    float* color;

    float* cudaDevicePosition;
    float* cudaDeviceVelField;
    float* cudaDeviceColor;
    float* cudaDeviceImageData;
    int* cudaDeviceLocks;

    List* cudaDeviceParticleList;

public:
    SimRenderer();
    virtual ~SimRenderer();

    const Image* getImage();

    void setup();

    void loadScene (Benchmark benchmark);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();
};
