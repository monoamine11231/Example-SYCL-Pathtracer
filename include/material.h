#ifndef PATHTRACER_INCLUDE_MATERIAL_H_
#define PATHTRACER_INCLUDE_MATERIAL_H_

#include <sycl/sycl.hpp>
#include "rapidobj.hpp"


struct Material {                       /* Filament standard model material props */
    sycl::vec<float, 3> baseColor;

    float               metallic;
    float               roughness;
    float               reflectance;

    float               emissive;       /* Luminocity */

    float               transmission;   /* Transparency */

    Material(sycl::vec<float, 3> baseColor,
             float metallic, float roughness, float reflectance,
             float emissive, float transmission)
        : baseColor(baseColor), metallic(metallic), roughness(roughness),
          reflectance(reflectance), emissive(emissive),
          transmission(transmission) {};

    Material(const rapidobj::Material& material);
};

#endif PATHTRACER_INCLUDE_MATERIAL_H_