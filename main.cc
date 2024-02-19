#include <iostream>
#include <optional>
#include <string>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <sycl/sycl.hpp>

#include "include/camera.h"
#include "include/object.h"
#include "include/ray.h"
#include "include/utils.h"

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


using Objects = std::variant<Sphere, Plane>;

const int kImageWidth = 1024;
const int kImageHeight = 512;

/* Antialiasing block */
const int kAABlockWidth = 2;
const int kAABlockHeight = 2;

const int kSamplesPerPixel = 1024;

const int kMaxRayDepth = 5;


Camera* camera_glb;
const float kCameraStep = 0.1f;


static void camera_keyback([[maybe_unused]] GLFWwindow *window, int key,
  [[maybe_unused]] int scancode, [[maybe_unused]] int action,
  [[maybe_unused]] int mods) {
  
  switch (key) {
  case GLFW_KEY_W:
    camera_glb->origin_ += camera_glb->GetFront() * kCameraStep;
    break;
  case GLFW_KEY_S:
    camera_glb->origin_ -= camera_glb->GetFront() * kCameraStep;
    break;
  case GLFW_KEY_D:
    camera_glb->origin_ += camera_glb->GetRight() * kCameraStep;
    break;
  case GLFW_KEY_A:
    camera_glb->origin_ -= camera_glb->GetRight() * kCameraStep;
    break;
  case GLFW_KEY_SPACE:
    camera_glb->origin_ += camera_glb->GetUp() * kCameraStep;
    break;
  case GLFW_KEY_LEFT_SHIFT:
    camera_glb->origin_ -= camera_glb->GetUp() * kCameraStep;
    break;
  default:
    break;
  }
}


int main() {
  GLFWwindow* window;

  /* Initialize the library */
  if (!glfwInit()) {
    printf("GLFW error: Could not initialize GLFW\n");
    return -1;
  }

  /* Create a windowed mode window and its OpenGL context */
  window = glfwCreateWindow(kImageWidth, kImageHeight, "SYCL Pathtracer", NULL, NULL);
  if (!window) {
    printf("GLFW error: Could not create a window\n");
    glfwTerminate();
    return -1;
  }

  /* Make the window's context current */
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, camera_keyback);
  glfwSwapInterval(1);
  
  GLenum glew_error = glewInit();
  if (glew_error != GLEW_OK) {
    printf("GLEW error: %s\n", glewGetErrorString(glew_error));
    return -1;
  }

  GLuint pbo, tex, fbo;
  /* Pixel buffer object initialization */
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_ARRAY_BUFFER, pbo);
  glBufferData(GL_ARRAY_BUFFER, kImageWidth*kImageHeight*3*4, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, kImageWidth, kImageHeight);
  // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, kImageWidth, kImageHeight, 0,  GL_RGB, GL_FLOAT, &ddd[0]);
  glBindTexture(GL_TEXTURE_2D, 0);

  /* Framebuffer object initialization */
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    printf("GL Framebuffer error: Could not attach GL texture as color attachment.\n");
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);


    /* Construct objects that are shared between host and device */
  sycl::device gpu(sycl::gpu_selector_v);
  sycl::queue q(gpu);

  CUstream custream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(q);

  cudaGraphicsResource *gresource;
  void *gresource_ptr;
  size_t gresource_size;
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&gresource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
  checkCudaErrors(cudaGraphicsMapResources(1, &gresource, custream));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&gresource_ptr, &gresource_size, gresource));

  auto* camera_mem = sycl::malloc_shared<Camera>(1, q);
  auto* container_mem =
      sycl::malloc_shared<containerutils::VariantContainer<Objects>>(1, q);

  /* Placement new to use allocated memory by SYCL to construct a class */
  Camera *camera =
      new (camera_mem) Camera(sycl::vec<float, 3>(1.0f, 0.0f, 0.0f),
                              sycl::vec<float, 3>(0.0f, 0.0f, 0.0f),
                              sycl::vec<float, 3>(0.0f, 0.0f, 1.0f), 90.0f,
                              1.0f, kImageWidth, kImageHeight);

  camera_glb = camera;

  containerutils::VariantContainer<Objects>* objects =
      new (container_mem) containerutils::VariantContainer<Objects>();

  objects->push_back(
      Sphere(
        sycl::vec<float, 3>(10.0f, 0.0f, 0.0f),
        2.0f, 0));

  objects->push_back(
      Sphere(
        sycl::vec<float, 3>(10.0f, 5.0f, 0.0f),
        1.0f, 0));

  objects->push_back(
      Plane(
        sycl::vec<float, 3>(10.0f, 0.0f, 0.0f),
        sycl::vec<float, 3>(0.05f, 0.0f, 2.0f),
        0));
  /* Path tracer program */
  auto pathtracer = [=](sycl::item<2> it) {
    auto w = it.get_id(0);
    auto h = it.get_id(1);

    sycl::device_ptr<float> image(reinterpret_cast<float*>(gresource_ptr));

    Ray global_ray;
    camera->GenerateRay(w, h, global_ray);

    for (uint32_t s = 0; s < kSamplesPerPixel; s++) {
      /* Copy the ray for each sample */
      Ray ray = global_ray;

        image[(kImageWidth*h+w)*3+2] = 1.0f;
        auto obj = closest_obj(ray, *objects);
        if (!obj.has_value()) {
          image[(kImageWidth*h+w)*3+0] = 0.0f;
          image[(kImageWidth*h+w)*3+1] = 0.0f;
          image[(kImageWidth*h+w)*3+2] = 0.0f;
          break;
        }
        image[(kImageWidth*h+w)*3+0] = 1.0f;
    }
  };




  glBindTexture(GL_TEXTURE_2D, tex);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  while (!glfwWindowShouldClose(window))
  {
      q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range(kImageWidth, kImageHeight), pathtracer);
      }).wait();

      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kImageWidth, kImageHeight, GL_RGB,  GL_FLOAT, NULL);


      glBlitFramebuffer(
          0, 0, kImageWidth, kImageHeight,
          0, 0, kImageWidth, kImageHeight,
          GL_COLOR_BUFFER_BIT, GL_LINEAR);



      /* Render here */

      /* Swap front and back buffers */
      glfwSwapBuffers(window);

      /* Poll for and process events */

      glfwPollEvents();
  }
  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

  checkCudaErrors(cudaGraphicsUnmapResources(1, &gresource, custream));
  glDeleteBuffers(1, &pbo);
  glDeleteTextures(1, &tex);
  glDeleteFramebuffers(1, &fbo);
  glfwTerminate();

  sycl::free(camera_mem, q);
  sycl::free(container_mem, q);

  return 0;
}