#include <iostream>
#include <optional>
#include <string>

#include <cmath>
#include <cstdint>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <sycl/sycl.hpp>

#include "include/camera.h"
#include "include/object.h"
#include "include/ray.h"
#include "include/material.h"
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
using Material = material::MicrofacetMaterial<material::FresnelSchlick,
  material::NormalGGX, material::GeometryGGXSchlick>;

const int kImageWidth = 1024;
const int kImageHeight = 512;

/* Antialiasing block */
const int kAABlockWidth = 2;
const int kAABlockHeight = 2;

const int kSamplesPerPixel = 1;

const int kMaxRayDepth = 5;


Camera* camera_glb;
int* executed_samples_glb;
int* total_executed_samples_glb;

/* Camera movement variables */
const float kCameraMoveStep = 0.1f;
const float kCameraLookStep = 0.02f;
float camera_xrot = 0;
float camera_yrot = M_PI_2;

static void camera_keyback([[maybe_unused]] GLFWwindow *window, int key,
  [[maybe_unused]] int scancode, [[maybe_unused]] int action,
  [[maybe_unused]] int mods) {
  
  *executed_samples_glb = 0;

  switch (key) {
  case GLFW_KEY_W:
    camera_glb->origin_ += camera_glb->GetFront() * kCameraMoveStep;
    break;
  case GLFW_KEY_S:
    camera_glb->origin_ -= camera_glb->GetFront() * kCameraMoveStep;
    break;
  case GLFW_KEY_D:
    camera_glb->origin_ += camera_glb->GetRight() * kCameraMoveStep;
    break;
  case GLFW_KEY_A:
    camera_glb->origin_ -= camera_glb->GetRight() * kCameraMoveStep;
    break;
  case GLFW_KEY_SPACE:
    camera_glb->origin_ += camera_glb->GetUp() * kCameraMoveStep;
    break;
  case GLFW_KEY_LEFT_SHIFT:
    camera_glb->origin_ -= camera_glb->GetUp() * kCameraMoveStep;
    break;
  
  case GLFW_KEY_UP:
    camera_yrot -= kCameraLookStep;
    break;
  case GLFW_KEY_DOWN:
    camera_yrot += kCameraLookStep;
    break;
  case GLFW_KEY_RIGHT:
    camera_xrot -= kCameraLookStep;
    break;
  case GLFW_KEY_LEFT:
    camera_xrot += kCameraLookStep;
    break;
  default:
    break;
  }

  sycl::vec<float, 3> ndir;
  ndir.x() = std::sin(camera_yrot)*std::cos(camera_xrot);
  ndir.y() = std::sin(camera_yrot)*std::sin(camera_xrot);
  ndir.z() = std::cos(camera_yrot);

  camera_glb->LookAt(ndir, sycl::vec<float,3>{0.0f,0.0f,1.0f});
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

  GLenum glew_error = glewInit();
  if (glew_error != GLEW_OK) {
    printf("GLEW error: %s\n", glewGetErrorString(glew_error));
    return -1;
  }

  GLuint pbo, tex, fbo;
  /* Pixel buffer object initialization */
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_ARRAY_BUFFER, pbo);
  glBufferData(GL_ARRAY_BUFFER, kImageWidth*kImageHeight*3, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, kImageWidth, kImageHeight);
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
  glFinish();

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

  /* SYCL memory allocation */
  Camera* camera = sycl::malloc_shared<Camera>(1, q);
  Material* materials = sycl::malloc_shared<Material>(4, q);
  containerutils::VariantContainer<Objects>* objects =
    sycl::malloc_shared<containerutils::VariantContainer<Objects>>(1, q);
  float* image = sycl::malloc_device<float>(kImageWidth*kImageHeight*3, q);
  int* executed_samples = sycl::malloc_shared<int>(1, q);
  int* total_executed_samples = sycl::malloc_shared<int>(1, q);

  /* Shared (unified) memory reflection to globals */
  camera_glb = camera;
  executed_samples_glb = executed_samples;
  total_executed_samples_glb = total_executed_samples;

  /* SYCL memory initialization */
  new (camera) Camera(sycl::vec<float, 3>(1.0f, 0.0f, 0.0f),
    sycl::vec<float, 3>(0.0f, 0.0f, 0.0f),
    sycl::vec<float, 3>(0.0f, 0.0f, 1.0f), 90.0f,
    1.0f, kImageWidth, kImageHeight);

  new (&materials[0]) Material(sycl::vec<float, 3>{0.0f,0.0f,1.0f}, 0.2f, 0.5f, false,
    0.0f, 0.0f);

  new (&materials[1]) Material(sycl::vec<float, 3>{4.0f,4.0f,4.0f}, 0.2f, 0.5f, false,
    0.0f, 0.0f);

  new (&materials[2]) Material(sycl::vec<float, 3>{1.0f,0.0f,0.0f}, 0.2f, 0.5f, false,
    0.0f, 0.0f);

  new (&materials[3]) Material(sycl::vec<float, 3>{0.0f,1.0f,0.0f}, 0.2f, 0.5f, false,
    0.0f, 0.0f);

  new (objects) containerutils::VariantContainer<Objects>();


  /* Filling the scene with objects */
  objects->push_back(
      Sphere(
        sycl::vec<float, 3>(10.0f, 0.0f, 0.0f),
        2.0f, 0));

  objects->push_back(
      Sphere(
        sycl::vec<float, 3>(10.0f, 5.0f, 0.0f),
        1.0f, 1));

  objects->push_back(
      Sphere(
        sycl::vec<float, 3>(7.0f, 0.0f, 0.0f),
        0.5f, 2));

  objects->push_back(
      Plane(
        sycl::vec<float, 3>(10.0f, 0.0f, -4.0f),
        sycl::vec<float, 3>(0.0f, 0.0f, 1.0f),
        0));
  objects->push_back(
      Plane(
        sycl::vec<float, 3>(15.0f, 0.0f, -4.0f),
        sycl::vec<float, 3>(-1.0f, 0.0f, 0.0f),
        3));


  /* Path tracer program */
  auto pathtracer = [=](sycl::nd_item<2> it) {
    auto w = it.get_global_id(0);
    auto h = it.get_global_id(1);

    sycl::device_ptr<uint8_t> framebuffer = reinterpret_cast<uint8_t*>(gresource_ptr);

    Ray global_ray, ray;
    camera->GenerateRay(w, h, global_ray);

    float &ir = image[(kImageWidth*h+w)*3+0];
    float &ig = image[(kImageWidth*h+w)*3+1];
    float &ib = image[(kImageWidth*h+w)*3+2];

    uint8_t &fr = framebuffer[(kImageWidth*h+w)*3+0];
    uint8_t &fg = framebuffer[(kImageWidth*h+w)*3+1]; 
    uint8_t &fb = framebuffer[(kImageWidth*h+w)*3+2];  

    if (*executed_samples == 0) {
      ir = 0.0f;
      ig = 0.0f;
      ib = 0.0f;
    }

    /* Good seed? */
    uint64_t seed = 0;
    seed |= (h & 0xFFFF) << 48;
    seed |= (w & 0xFFFF) << 32;
    seed |= *total_executed_samples & 0xFFFFFFFF;
    /* If seed not hashed, artifacts appear */
    miscutils::XorShiftPRNG random(miscutils::Hash64(seed));

    for (int s = 0; s < kSamplesPerPixel; s++) {
      ray = global_ray;
      float mu = 1.0f;
      while (ray.depth < kMaxRayDepth) {
        auto obj = closest_obj(ray, *objects);
        if (!obj.has_value()) {
          ir += mu*0.6f;
          ig += mu*0.6f;
          ib += mu*0.6f;
          break;
        }

        const Intersector &intersection = *obj;

        const sycl::vec<float, 3> &n = intersection.normal;
        Material &material = materials[intersection.material_id];
        sycl::vec<float,3> f = mu*material.base_color;
        ir += f.x();
        ig += f.y();
        ib += f.z();

        
        sycl::vec<float, 3> h, l;
        material.Sample(random, ray.dir, n, h, l);
        // printf("n:%f,%f,%f\n", h.x(), h.y(), h.z());
        // mu *= (0.8f*M_1_PI+0.2f*material.Eval(l, ray.dir, n, h))*sycl::clamp(sycl::dot(n,l), 0.0f, 1.0f);
        mu *= 2 * 0.18f * sycl::dot(n,l);
        ray.depth += 1;
        ray.origin += ray.dir * intersection.t;
        ray.origin += intersection.normal*0.1f;
        ray.dir = l;
      }
    }

    /* Gamma correction */
    float kGamma = 1.0f/2.2f;
    fr = sycl::pow(sycl::clamp(ir/(*executed_samples+1),0.0f,1.0f),kGamma)*255;
    fg = sycl::pow(sycl::clamp(ig/(*executed_samples+1),0.0f,1.0f),kGamma)*255;
    fb = sycl::pow(sycl::clamp(ib/(*executed_samples+1),0.0f,1.0f),kGamma)*255;
  };




  
  while (!glfwWindowShouldClose(window))
  {
      /* NOTE here how `sycl::nd_range` is used instead of `sycl::range`. This part is
       * is important since it specifies the work group size. If the work group is not
       * specified explicitely, this may result in bugs where the workers process data
       * outside of the given range!!!!!!!
       */
      q.submit([&](sycl::handler& h) {
        sycl::range<2> global_range{kImageWidth, kImageHeight};
        sycl::range<2> local_range{kAABlockWidth, kAABlockHeight};
        h.parallel_for(sycl::nd_range{global_range,local_range}, pathtracer);
      }).wait_and_throw();
      *executed_samples_glb += kSamplesPerPixel;
      *total_executed_samples_glb += kSamplesPerPixel;
      // printf("Done rendering frame\n");

      glBindTexture(GL_TEXTURE_2D, tex);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kImageWidth, kImageHeight, GL_RGB,
        GL_UNSIGNED_BYTE, NULL);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glBindTexture(GL_TEXTURE_2D, 0);


      glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

      glBlitFramebuffer(
          0, 0, kImageWidth, kImageHeight,
          0, 0, kImageWidth, kImageHeight,
          GL_COLOR_BUFFER_BIT, GL_NEAREST);

      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);


      /* Swap front and back buffers */
      glfwSwapBuffers(window);

      glfwPollEvents();
  }
  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

  checkCudaErrors(cudaGraphicsUnmapResources(1, &gresource, custream));
  glDeleteBuffers(1, &pbo);
  glDeleteTextures(1, &tex);
  glDeleteFramebuffers(1, &fbo);
  glfwTerminate();

  sycl::free(camera, q);
  sycl::free(materials, q);
  sycl::free(objects, q);
  sycl::free(image, q);
  sycl::free(executed_samples, q);
  sycl::free(total_executed_samples, q);

  return 0;
}