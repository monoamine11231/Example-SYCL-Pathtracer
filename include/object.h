#ifndef PATHTRACER_INCLUDE_OBJECT_H_
#define PATHTRACER_INCLUDE_OBJECT_H_

#include <variant>

#include "objects/mesh.h"
#include "objects/plane.h"
#include "objects/sphere.h"
#include "ray.h"
#include "variantcontainer.h"

using obj = std::variant<Mesh, Sphere, Plane>;

std::optional<Intersector> closest_obj(const Ray &ray,
                                       const VariantContainer<obj> &objects);

#endif