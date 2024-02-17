#include <iostream>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>

#include "include/camera.h"
#include "include/object.h"
#include "include/ray.h"
#include "include/utils.h"

using Pixel = sycl::vec<float, 3>;
using Objects = std::variant<Sphere, Plane>;


const uint16_t kImageWidth = 1024;
const uint16_t kImageHeight = 512;

/* Antialiasing block */
const uint8_t kAABlockWidth = 2;
const uint8_t kAABlockHeight = 2;

const uint32_t kSamplesPerPixel = 1024;

const uint8_t kMaxRayDepth = 5;

int main() {
  // using var = std::variant<int, float>;

  // VariantContainer<var> container{};
  // container.push_back(12);
  // container.push_back(22);
  // container.push_back(32);
  // container.push_back(42);

  // container.push_back(12.2f);
  // container.push_back(22.5f);
  // container.push_back(32.22f);
  // container.push_back(42.555f);

  // float sum = 0.0f;
  // auto a = [&sum](const auto &a) { sum += a; };

  // container.forEach(a);

  // std::cout << sum << std::endl;

  /* Construct objects that are shared between host and device */
  sycl::queue q(sycl::gpu_selector_v);

  auto* camera_mem = sycl::malloc_shared<Camera>(1, q);
  auto* container_mem =
      sycl::malloc_shared<containerutils::VariantContainer<Objects>>(1, q);

  /* Placement new to use allocated memory by SYCL to construct a class */
  Camera* camera =
      new (camera_mem) Camera(sycl::vec<float, 3>(1.0f, 0.0f, 0.0f),
                              sycl::vec<float, 3>(0.0f, 0.0f, 0.0f),
                              sycl::vec<float, 3>(0.0f, 0.0f, 1.0f), 90.0f,
                              1.0f, kImageWidth, kImageHeight);
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
  Pixel* image = sycl::malloc_shared<Pixel>(kImageWidth * kImageHeight, q);
  /* Placing everything in a scope forces host to wait for everything to
   * complete */
  {


    /* Path tracer program */
    auto pathtracer = [=](sycl::item<2> it) {
      auto w = it.get_id(0);
      auto h = it.get_id(1);

      Ray global_ray;
      camera->GenerateRay(w, h, global_ray);

      for (uint32_t s = 0; s < kSamplesPerPixel; s++) {
        /* Copy the ray for each sample */
        Ray ray = global_ray;

          image[kImageWidth*h+w].z() = 1.0f;
          auto obj = closest_obj(ray, *objects);
          if (!obj.has_value()) break;
          image[kImageWidth*h+w].x() = 1.0f;
      }
    };
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range(kImageWidth, kImageHeight), pathtracer);
    }).wait();
  }

  printf("P3\n%d %d\n255\n", kImageWidth, kImageHeight);
  for (int y = kImageHeight - 1; y >= 0; --y) {
    for (int x = 0; x < kImageWidth; ++x) {
      int id = y*kImageWidth+x,r,g,b;
      r = static_cast<int>(255.99*image[id].x());
      g = static_cast<int>(255.99*image[id].y());
      b = static_cast<int>(255.99*image[id].z());

      printf("%d %d %d\n",r,g,b);
    }
  }

  sycl::free(camera_mem, q);
  sycl::free(container_mem, q);
  sycl::free(image, q);

  return 0;
}