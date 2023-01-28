#ifndef PATHTRACER_INCLUDE_RAY_H_
#define PATHTRACER_INCLUDE_RAY_H_

#include <sycl/sycl.hpp>


struct Ray {
    sycl::vec<float, 3>     origin;
    sycl::vec<float, 3>     dir;        /* Normalized direction vector */

    Ray(sycl::vec<float, 3> origin, sycl::vec<float, 3> dir)
        : origin(origin), dir(dir) {};
};


struct Intersector {
    float                   t;
    sycl::vec<float, 3>     normal;
    uint8_t                 material_id;

    Intersector(float t, sycl::vec<float, 3> normal, uint8_t material_id)
        : t(t), normal(normal), material_id(material_id) {};
};

#endif