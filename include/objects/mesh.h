#ifndef PATHTRACER_INCLUDE_OBJECTS_MESH_H_
#define PATHTRACER_INCLUDE_OBJECTS_MESH_H_

#include <sycl/sycl.hpp>
#include "ray.h"


class MeshTriangle {
private:
    sycl::vec<float, 3> a_;
    sycl::vec<float, 3> b_;
    sycl::vec<float, 3> c_;

    sycl::vec<float, 3> normal_;
public:

    MeshTriangle(sycl::vec<float, 3> a, sycl::vec<float, 3> b, sycl::vec<float, 3> c,
                 sycl::vec<float, 3> normal)
        : a_(a), b_(b), c_(c), normal_(normal) {};

    std::optional<Intersector> intersect(const Ray& ray);
};

class Mesh{
private:
    std::vector<MeshTriangle>   faces_;

    uint8_t                     material_id_;
public:

    Mesh(uint8_t material_id) : material_id_(material_id) {};
};



#endif