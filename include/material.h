#ifndef PATHTRACER_INCLUDE_MATERIAL_H_
#define PATHTRACER_INCLUDE_MATERIAL_H_

#include <sycl/sycl.hpp>

#include "rapidobj/rapidobj.hpp"

namespace material {
/* Function objects */
struct FresnelSchlick {
  FresnelSchlick(){};

  /* `n` is the halfway vector when applying the Microfacet model */
  float dielectric(float f0, const sycl::vec<float, 3> &n,
                   const sycl::vec<float, 3> &v);
};

struct NormalGGX {
  NormalGGX(){};
  float operator()(float roughness, const sycl::vec<float, 3> &n,
                   const sycl::vec<float, 3> &h);
};

struct GeometryGGXSchlick {
  GeometryGGXSchlick(){};
  float operator()(float roughness, const sycl::vec<float, 3> &n,
                   const sycl::vec<float, 3> &v);
};

/* Microfacet material model */
template <class Fresnel, class Normal, class Geometry>
struct MicrofacetMaterial {
  Fresnel fresnel;
  Normal normal;
  Geometry geometry;

  sycl::vec<float, 3> base_color;

  float metallic;
  float roughness;

  bool dielectric;
  float reflectance;

  float emitance;

  MicrofacetMaterial(const rapidobj::Material &material);
  MicrofacetMaterial(sycl::vec<float, 3> base_color, float metallic,
                     float roughness, bool dielectric, float reflectance,
                     float emmitance)
      : fresnel(Fresnel()),
        normal(Normal()),
        geometry(Geometry()),
        base_color(base_color),
        metallic(metallic),
        roughness(roughness),
        dielectric(dielectric),
        reflectance(reflectance),
        emitance(emmitance){};
};

} /* namespace material */

#endif