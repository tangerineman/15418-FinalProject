struct Image;

typedef enum {
    SIMPLE,
    COMPLEX
} Benchmark;

class SimRenderer {
private:
    Image* image;
    Benchmark benchmark;

    int numberOfParticles;

    float* position;
    float* velField;
    float* color;

    float* cudaDevicePosition;
    float* cudaDeviceVelField;
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
