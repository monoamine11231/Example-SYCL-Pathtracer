#ifndef PATHTRACER_INCLUDE_OBJECTS_SPHERE_H_
#define PATHTRACER_INCLUDE_OBJECTS_SPHERE_H_

#include <sycl/sycl.hpp>

#include "include/ray.h"

class Sphere {
 private:
  sycl::vec<float, 3> origin_;
  float radius_;

  uint8_t material_id_;

 public:
  SYCL_EXTERNAL Sphere(sycl::vec<float, 3> origin, float radius, uint8_t material_id)
      : origin_(origin), radius_(radius), material_id_(material_id){};

  SYCL_EXTERNAL std::optional<Intersector> Intersect(const Ray& ray) const;
};

#endif