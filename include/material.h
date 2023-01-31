#ifndef PATHTRACER_INCLUDE_MATERIAL_H_
#define PATHTRACER_INCLUDE_MATERIAL_H_

#include <sycl/sycl.hpp>
#include "rapidobj/rapidobj.hpp"


/* Strauss reflectance model */
struct StraussMaterial {
    sycl::vec<float, 3> color;

    float               diffuse;
    float               specular;

    float               smoothness;
    float               metalness;
    float               transparency;
    
    float               ior;

    StraussMaterial(const rapidobj::Material& material);
};

#endif PATHTRACER_INCLUDE_MATERIAL_H_