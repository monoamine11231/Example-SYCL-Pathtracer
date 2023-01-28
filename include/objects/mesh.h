#ifndef PATHTRACER_INCLUDE_OBJECTS_MESH_H_
#define PATHTRACER_INCLUDE_OBJECTS_MESH_H_

#include <sycl/sycl.hpp>
#include "rapidobj/rapidobj.hpp"
#include "include/ray.h"


class MeshTriangle {
private:
    sycl::vec<float, 3> a_;
    sycl::vec<float, 3> b_;
    sycl::vec<float, 3> c_;

    sycl::vec<float, 3> normal_;

    uint8_t             material_id_;
public:

    MeshTriangle(sycl::vec<float, 3> a, sycl::vec<float, 3> b, sycl::vec<float, 3> c,
                 sycl::vec<float, 3> normal, uint8_t material_id)
        : a_(a), b_(b), c_(c), normal_(normal), material_id_(material_id) {};

    std::optional<Intersector> Intersect(const Ray& ray) const;
};

class Mesh{

public:
    std::vector<MeshTriangle>   faces_;

    Mesh(const rapidobj::Shape& shape, const rapidobj::Attributes& attributes);

    std::optional<Intersector> Intersect(const Ray& ray) const;
};

#endif