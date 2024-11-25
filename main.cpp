#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <GL/glew.h>
#include <GLFW/glfw3.h> // GLFW is still used for OpenGL context management
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cstdlib> // For rand()
#include <cstring>
#include "stb_image_write.h"

const int IMAGE_WIDTH = 800;
const int IMAGE_HEIGHT = 800;
const int NUM_POINTS = 1000; // Number of points to render
const float POINT_RADIUS = 10.0f; // Radius of the circles in pixels

struct Point {
    float x, y;
};

struct RGB {
    unsigned char r, g, b;
};

Point *cudaPoints;
RGB *cudaColors;
cudaGraphicsResource *cudaVbo, *cudaColorVbo;
GLuint vbo, colorVbo, fbo, renderTexture;

void createRandomPoints(std::vector<Point>& points, int numPoints) {
    for (int i = 0; i < numPoints; ++i) {
        points[i].x = -1.0f + 2.0f * ((float)rand() / RAND_MAX); // Random x in [-1, 1]
        points[i].y = -1.0f + 2.0f * ((float)rand() / RAND_MAX); // Random y in [-1, 1]
    }
}

RGB scalarToRGB(float scalar) {
    unsigned char r = (unsigned char)(255 * scalar);
    unsigned char g = 0;
    unsigned char b = (unsigned char)(255 * (1.0f - scalar));
    return {r, g, b};
}

void createRandomColors(std::vector<RGB>& colors, int numPoints) {
    for (int i = 0; i < numPoints; ++i) {
        float scalar = (float)rand() / RAND_MAX; // Random scalar in [0, 1]
        colors[i] = scalarToRGB(scalar);
    }
}

void initVbos(int numPoints) {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numPoints * sizeof(Point), nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &colorVbo);
    glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
    glBufferData(GL_ARRAY_BUFFER, numPoints * sizeof(RGB), nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cudaVbo, vbo, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cudaColorVbo, colorVbo, cudaGraphicsMapFlagsWriteDiscard);
}

void initFramebuffer() {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &renderTexture);
    glBindTexture(GL_TEXTURE_2D, renderTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTexture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer is incomplete!" << std::endl;
        exit(-1);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void copyHostToDevice(const std::vector<Point>& points, const std::vector<RGB>& colors, int numPoints) {
    Point *mappedPoints = nullptr;
    RGB *mappedColors = nullptr;

    cudaGraphicsMapResources(1, &cudaVbo, 0);
    cudaGraphicsMapResources(1, &cudaColorVbo, 0);

    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void **)&mappedPoints, &numBytes, cudaVbo);
    cudaGraphicsResourceGetMappedPointer((void **)&mappedColors, &numBytes, cudaColorVbo);

    cudaMemcpy(mappedPoints, points.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(mappedColors, colors.data(), numPoints * sizeof(RGB), cudaMemcpyHostToDevice);

    cudaGraphicsUnmapResources(1, &cudaVbo, 0);
    cudaGraphicsUnmapResources(1, &cudaColorVbo, 0);
}

void render(int numPoints) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);

    glClear(GL_COLOR_BUFFER_BIT);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, sizeof(Point), (void *)0);

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

void saveFramebufferToPNG(const char *filename) {
    std::vector<unsigned char> pixels(IMAGE_WIDTH * IMAGE_HEIGHT * 3);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    std::vector<unsigned char> flippedPixels(IMAGE_WIDTH * IMAGE_HEIGHT * 3);
    for (int y = 0; y < IMAGE_HEIGHT; ++y) {
        std::memcpy(&flippedPixels[y * IMAGE_WIDTH * 3],
                    &pixels[(IMAGE_HEIGHT - 1 - y) * IMAGE_WIDTH * 3],
                    IMAGE_WIDTH * 3);
    }

    if (stbi_write_png(filename, IMAGE_WIDTH, IMAGE_HEIGHT, 3, flippedPixels.data(), IMAGE_WIDTH * 3)) {
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

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Invisible window
    GLFWwindow *window = glfwCreateWindow(IMAGE_WIDTH, IMAGE_HEIGHT, "Off-Screen", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW context\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

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

