#ifndef PATHTRACER_INCLUDE_OBJECT_H_
#define PATHTRACER_INCLUDE_OBJECT_H_

#include <variant>

#include "objects/mesh.h"
#include "objects/sphere.h"
#include "objects/plane.h"


using obj = std::variant<Mesh, Sphere, Plane>;

#endif