#ifndef PATHTRACER_INCLUDE_OBJECT_H_
#define PATHTRACER_INCLUDE_OBJECT_H_

#include <variant>

// #include "objects/mesh.h"
#include "objects/plane.h"
#include "objects/sphere.h"
#include "include/ray.h"
#include "include/utils.h"

using Objects = std::variant<Sphere, Plane>;

SYCL_EXTERNAL std::optional<Intersector> closest_obj(
    const Ray &ray, const containerutils::VariantContainer<Objects> &objects);

#endif