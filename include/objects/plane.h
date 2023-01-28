#ifndef PATHTRACER_INCLUDE_OBJECTS_PLANE_H_
#define PATHTRACER_INCLUDE_OBJECTS_PLANE_H_

#include <sycl/sycl.hpp>
#include "ray.h"


class Plane {
private:
    sycl::vec<float, 3>     point_;     /* Point in plane */
    sycl::vec<float, 3>     normal_;

    uint8_t                 material_id_;
public:

    Plane(sycl::vec<float, 3> point, sycl::vec<float, 3> normal, uint8_t material_id)
        : point_(point), normal_(normal), material_id_(material_id) {};

    std::optional<Intersector> intersect(const Ray& ray);
};

#endif