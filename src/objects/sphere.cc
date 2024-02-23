#include "include/objects/sphere.h"

#include <sycl/sycl.hpp>

std::optional<Intersector> Sphere::Intersect(const Ray& ray) const {
  std::optional<Intersector> intersection{};

  sycl::vec<float, 3> v, normal;
  float a, b, c, D, t;

  v = ray.origin - this->origin_;

  a = sycl::dot(ray.dir, ray.dir);
  b = sycl::dot(2.0f * v, ray.dir);
  c = sycl::dot(v, v) - this->radius_ * this->radius_;

  D = b * b - 4 * a * c;

  if (D < 0.0f) {
    /* If determinant = 0.0f, return an empty intersection */
    return intersection;
  }

  D = sycl::sqrt(D);

  /*  If the closest intersection is behind the ray's origin replace it with
      ray-sphere intersection point infront of the ray's origin */
  t = ((-b - D) / (2.0f * a) < 0) ? (-b + D) / (2.0f * a)
                                  : (-b - D) / (2.0f * a);
  if (t <= 0.0f) {
    /*  If the intersection point is still behind the ray's origin,
        return an empty intersection */
    return intersection;
  }

  normal = ray.origin + t * ray.dir - this->origin_;
  normal = sycl::normalize(normal);

  Intersector data(t, normal, this->material_id_);
  intersection = data;

  return intersection;
}