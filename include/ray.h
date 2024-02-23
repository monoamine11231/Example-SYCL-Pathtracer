#ifndef PATHTRACER_INCLUDE_RAY_H_
#define PATHTRACER_INCLUDE_RAY_H_

#include <sycl/sycl.hpp>

struct Ray {
  sycl::vec<float, 3> origin;
  sycl::vec<float, 3> dir; /* Normalized direction vector */

  int depth;

  SYCL_EXTERNAL Ray(sycl::vec<float, 3> origin, sycl::vec<float, 3> dir)
      : origin(origin), dir(dir), depth(0){};

  SYCL_EXTERNAL Ray() : depth(0) {};
};

struct Intersector {
  float t;
  sycl::vec<float, 3> normal;
  uint8_t material_id;

  SYCL_EXTERNAL Intersector(float t, sycl::vec<float, 3> normal, uint8_t material_id)
      : t(t), normal(normal), material_id(material_id){};
};

#endif