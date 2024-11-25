#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <iostream>
#include <vector>
#include <cstring> // For memcpy

const int IMAGE_WIDTH = 800;
const int IMAGE_HEIGHT = 800;

void renderScene() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // White background
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Enable smooth point rendering
    glEnable(GL_POINT_SMOOTH);

    glPointSize(10.0f); // Set point size
    glBegin(GL_POINTS);
    for (int i = 0; i < 1000; ++i) {
        float x = -1.0f + 2.0f * ((float)rand() / RAND_MAX); // Random x in [-1, 1]
        float y = -1.0f + 2.0f * ((float)rand() / RAND_MAX); // Random y in [-1, 1]
        glColor4f((float)rand() / RAND_MAX,                // Random red
                  (float)rand() / RAND_MAX,                // Random green
                  (float)rand() / RAND_MAX,                // Random blue
                  0.7f);                                   // Semi-transparent alpha
        glVertex2f(x, y);
    }
    glEnd();

    glFlush();
}

void saveJPG(const char *filename, const unsigned char *buffer, int width, int height) {
    // Convert RGBA to RGB by discarding the alpha channel
    std::vector<unsigned char> rgbBuffer(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        rgbBuffer[i * 3 + 0] = buffer[i * 4 + 0]; // R
        rgbBuffer[i * 3 + 1] = buffer[i * 4 + 1]; // G
        rgbBuffer[i * 3 + 2] = buffer[i * 4 + 2]; // B
    }

    // Flip the image vertically because OpenGL's origin is bottom-left
    std::vector<unsigned char> flippedBuffer(width * height * 3);
    for (int y = 0; y < height; ++y) {
        std::memcpy(&flippedBuffer[y * width * 3],
                    &rgbBuffer[(height - 1 - y) * width * 3],
                    width * 3);
    }

    // Save as JPG
    if (stbi_write_jpg(filename, width, height, 3, flippedBuffer.data(), 100)) {
        std::cout << "Image saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image!" << std::endl;
    }
}

int main() {
    // Initialize EGL
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        std::cerr << "Failed to get EGL display!" << std::endl;
        return -1;
    }

    if (!eglInitialize(display, nullptr, nullptr)) {
        std::cerr << "Failed to initialize EGL!" << std::endl;
        return -1;
    }

    // Configure EGL
    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE
    };

    EGLConfig config;
    EGLint numConfigs;
    if (!eglChooseConfig(display, configAttribs, &config, 1, &numConfigs) || numConfigs < 1) {
        std::cerr << "Failed to choose EGL config!" << std::endl;
        return -1;
    }

    // Create an off-screen surface
    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, IMAGE_WIDTH,
        EGL_HEIGHT, IMAGE_HEIGHT,
        EGL_NONE
    };
    EGLSurface surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
    if (surface == EGL_NO_SURFACE) {
        std::cerr << "Failed to create EGL surface!" << std::endl;
        return -1;
    }

    // Bind OpenGL API
    if (!eglBindAPI(EGL_OPENGL_API)) {
        std::cerr << "Failed to bind OpenGL API!" << std::endl;
        return -1;
    }

    // Create an EGL context
    const EGLint contextAttribs[] = {EGL_NONE};
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
        std::cerr << "Failed to create EGL context!" << std::endl;
        return -1;
    }

    // Make the context current
    if (!eglMakeCurrent(display, surface, surface, context)) {
        std::cerr << "Failed to make EGL context current!" << std::endl;
        return -1;
    }

    // Allocate memory for the framebuffer
    std::vector<unsigned char> framebuffer(IMAGE_WIDTH * IMAGE_HEIGHT * 4); // RGBA format

    // Set the viewport
    glViewport(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);

    // Render the scene
    renderScene();

    // Read pixels from the framebuffer
    glReadPixels(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer.data());

    // Save the framebuffer to a JPG file
    saveJPG("output.jpg", framebuffer.data(), IMAGE_WIDTH, IMAGE_HEIGHT);

    // Clean up
    eglDestroySurface(display, surface);
    eglDestroyContext(display, context);
    eglTerminate(display);

    return 0;
}
