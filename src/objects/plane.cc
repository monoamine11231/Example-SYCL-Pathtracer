#include "include/objects/plane.h"

#include <sycl/sycl.hpp>

std::optional<Intersector> Plane::Intersect(const Ray& ray) const {
  std::optional<Intersector> intersection;
  float determinant, t;

  determinant = sycl::dot(this->normal_, ray.dir);
  if (determinant == 0.0f) {
    /* If the ray and plane are paralell, return empty intersection */
    return intersection;
  }

  t = sycl::dot(this->point_ - ray.origin, this->normal_) / determinant;
  if (t <= 0.0f) {
    /*  If the intersection point is on the opposite of ray direction
        return an empty intersection. */
    return intersection;
  }

  Intersector data(t, this->normal_, this->material_id_);
  intersection = data;
  return intersection;
}