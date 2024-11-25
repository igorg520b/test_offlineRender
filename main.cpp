#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cstdlib> // For rand()
#include <cstring>
#include "stb_image_write.h"


const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;
const int NUM_POINTS = 1000; // Number of points to render
const double POINT_RADIUS = 10.0; // Radius of the circles in pixels

struct Point {
    double x, y;
};

struct RGB {
    unsigned char r, g, b;
};

Point *cudaPoints;
RGB *cudaColors;
cudaGraphicsResource *cudaVbo, *cudaColorVbo;
GLuint vbo, colorVbo, fbo, renderTexture;

// Function to create random points
void createRandomPoints(std::vector<Point>& points, int numPoints) {
    for (int i = 0; i < numPoints; ++i) {
        points[i].x = -1.0 + 2.0 * ((double)rand() / RAND_MAX); // Random x in [-1, 1]
        points[i].y = -1.0 + 2.0 * ((double)rand() / RAND_MAX); // Random y in [-1, 1]
    }
}

// Custom scalar-to-RGB conversion function
RGB scalarToRGB(float scalar) {
    unsigned char r = (unsigned char)(255 * scalar);
    unsigned char g = 0;
    unsigned char b = (unsigned char)(255 * (1.0f - scalar));
    return {r, g, b};
}

// Function to create random colors
void createRandomColors(std::vector<RGB>& colors, int numPoints) {
    for (int i = 0; i < numPoints; ++i) {
        float scalar = (float)rand() / RAND_MAX; // Random scalar in [0, 1]
        colors[i] = scalarToRGB(scalar);
    }
}

// Initialize OpenGL VBOs
void initVbos(int numPoints) {
    // Points VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numPoints * sizeof(Point), nullptr, GL_DYNAMIC_DRAW);

    // Colors VBO
    glGenBuffers(1, &colorVbo);
    glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
    glBufferData(GL_ARRAY_BUFFER, numPoints * sizeof(RGB), nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBOs with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaVbo, vbo, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cudaColorVbo, colorVbo, cudaGraphicsMapFlagsWriteDiscard);
}

// Initialize the framebuffer for off-screen rendering
void initFramebuffer() {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Create texture to store the framebuffer image
    glGenTextures(1, &renderTexture);
    glBindTexture(GL_TEXTURE_2D, renderTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTexture, 0);

    // Check if framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Failed to create framebuffer!" << std::endl;
        exit(-1);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Copy host data to CUDA buffers
void copyHostToDevice(const std::vector<Point>& points, const std::vector<RGB>& colors, int numPoints) {
    // Map OpenGL buffers for CUDA
    Point *mappedPoints = nullptr;
    RGB *mappedColors = nullptr;

    cudaGraphicsMapResources(1, &cudaVbo, 0);
    cudaGraphicsMapResources(1, &cudaColorVbo, 0);

    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void **)&mappedPoints, &numBytes, cudaVbo);
    cudaGraphicsResourceGetMappedPointer((void **)&mappedColors, &numBytes, cudaColorVbo);

    // Copy data to CUDA buffers
    cudaMemcpy(mappedPoints, points.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(mappedColors, colors.data(), numPoints * sizeof(RGB), cudaMemcpyHostToDevice);

    // Unmap resources
    cudaGraphicsUnmapResources(1, &cudaVbo, 0);
    cudaGraphicsUnmapResources(1, &cudaColorVbo, 0);
}

// Render points with colors
void render(int numPoints) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    glClear(GL_COLOR_BUFFER_BIT);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_DOUBLE, sizeof(Point), (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
    glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(RGB), (void *)0);

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPointSize(POINT_RADIUS);
    glDrawArrays(GL_POINTS, 0, numPoints);

    glDisable(GL_BLEND);
    glDisable(GL_POINT_SMOOTH);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Save the rendered image to PNG
void saveFramebufferToPNG(const char *filename) {
    std::vector<unsigned char> pixels(WINDOW_WIDTH * WINDOW_HEIGHT * 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Flip the image vertically because OpenGL's origin is bottom-left
    std::vector<unsigned char> flippedPixels(WINDOW_WIDTH * WINDOW_HEIGHT * 3);
    for (int y = 0; y < WINDOW_HEIGHT; ++y) {
        std::memcpy(&flippedPixels[y * WINDOW_WIDTH * 3],
                    &pixels[(WINDOW_HEIGHT - 1 - y) * WINDOW_WIDTH * 3],
                    WINDOW_WIDTH * 3);
    }

    if (stbi_write_png(filename, WINDOW_WIDTH, WINDOW_HEIGHT, 3, flippedPixels.data(), WINDOW_WIDTH * 3)) {
        std::cout << "Image saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image!" << std::endl;
    }
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA + OpenGL (Save to PNG)", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

    glEnable(GL_MULTISAMPLE);

    std::vector<Point> points(NUM_POINTS);
    createRandomPoints(points, NUM_POINTS);

    std::vector<RGB> colors(NUM_POINTS);
    createRandomColors(colors, NUM_POINTS);

    initVbos(NUM_POINTS);
    initFramebuffer();
    copyHostToDevice(points, colors, NUM_POINTS);

    render(NUM_POINTS);
    saveFramebufferToPNG("output.png");

    cudaGraphicsUnregisterResource(cudaVbo);
    cudaGraphicsUnregisterResource(cudaColorVbo);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &colorVbo);
    glDeleteTextures(1, &renderTexture);
    glDeleteFramebuffers(1, &fbo);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
