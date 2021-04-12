struct Image;

typedef enum {
    SIMPLE,
    COMPLEX
} Benchmark;

class SimRenderer {
private:
    Image* image;
    Benchmark benchmark;

    int numParticles;

    float* position;
    float* velocity;
    float* color;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceImageData;
    int* cudaDeviceLocks;

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
