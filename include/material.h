#ifndef PATHTRACER_INCLUDE_MATERIAL_H_
#define PATHTRACER_INCLUDE_MATERIAL_H_

#include <cmath>
#include <sycl/sycl.hpp>

#include "include/utils.h"

/* For debugging */
const bool kUseAngleForSampling = false;

namespace material {
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

  float fresnel0;

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

  /* Samples the halfway vector and the new direction vector from geometric
     normal and the view direction. `h` and `l` are overwritten by the sampled
     halfway vector and the new direction vector */
  template <class Random>
  void Sample(Random &random, const sycl::vec<float, 3> &v,
              const sycl::vec<float, 3> &n, sycl::vec<float, 3> &h,
              sycl::vec<float, 3> &l) {
    this->normal.SampleHVec(*this, random(), random(), n, h);
    /* Sampled outgoing direction (light direction) */
    l = 2 * sycl::dot(v, h) * h - v;
  }

  /* Returns the BRDF/PDF from the given new direction vector `l`, view vector
   * `v`, geometric normal `n` and halfway vector `h`*/
  template <class Random>
  float Eval(const sycl::vec<float, 3> &l, const sycl::vec<float, 3> &v,
             const sycl::vec<float, 3> &n, const sycl::vec<float, 3> &h) {
    /* Fresnel term in BRDF */
    sycl::vec<float, 3> F = this->fresnel(*this, h, l);
    /* Geometric term in BRDF */
    float G = this->geometry(*this, l, v, h);

    float nv_dot = sycl::clamp(sycl::dot(n, v), 0.0f, 1.0f);
    float hn_dot = sycl::clamp(sycl::dot(h, n), 0.0f, 1.0f);
    /* Denominator term */
    float denominator = sycl::clamp(sycl::dot(v, n), 0.0f, 1.0f) *
                        sycl::clamp(sycl::dot(h, n), 0.0f, 1.0f);

    return F.s0() * G * sycl::clamp(sycl::dot(v, h), 0.0f, 1.0f) / denominator;
  }
};

/* Function objects */
class FresnelSchlick {
  FresnelSchlick(){};

  /* One fresnel for both dielectric and metallic materials */
  /* `n` is the halfway vector when applying the Microfacet model */
  template <class Fresnel, class Normal, class Geometry>
  sycl::vec<float, 3> operator()(
      const MicrofacetMaterial<Fresnel, Normal, Geometry> &material,
      const sycl::vec<float, 3> &n, const sycl::vec<float, 3> &v) {
    sycl::vec<float, 3> F0{material.fresnel0, material.fresnel0,
                           material.fresnel0};

    /* Linear mix f0 and base color based on metaliness */
    F0 = vecutils::Lerp(F0, material.base_color, material.metallic);

    sycl::vec<float, 3> ones{1.0f, 1.0f, 1.0f};
    float rdot = 1 - sycl::dot(n, v);

    return F0 + (ones - F0) * (rdot * rdot * rdot * rdot * rdot);
  }
};

class NormalGGX {
  NormalGGX(){};
  /* `n` is the halfway vector when applying the Microfacet model */
  template <class Fresnel, class Geometry>
  float operator()(
      const MicrofacetMaterial<Fresnel, material::NormalGGX, Geometry> &material,
      const sycl::vec<float, 3> &n, const sycl::vec<float, 3> &h) {
    float alpha = material.roughness * material.roughness;
    float alpha_sq = alpha * alpha;
    float dot = sycl::dot(n, h);

    float tmp = dot * dot * alpha_sq + (1 - dot * dot);
    return (miscutils::ShadowFactor(dot) * alpha_sq) / (M_PI * tmp * tmp);
  }

  /* Returns whole angles for theta and phi */
  template <class Fresnel, class Geometry>
  std::pair<float, float> Sample(
      const MicrofacetMaterial<Fresnel, material::NormalGGX, Geometry> &material,
      float u1, float u2) {
    float alpha = material.roughness * material.roughness;

    /* Sampling of normal distribution function */
    float theta = atan((alpha * sqrt(u1)) / sqrt(1 - u1));
    float phi = 2 * M_PI * u2;

    return std::pair<float, float>(theta, phi);
  }

  /* Returns theta in cos and sin, while phi is returned as an angle. It is more
   * efficient to compute this */
  template <class Fresnel, class Geometry>
  std::tuple<float, float, float> SampleTrigonometric(
      const MicrofacetMaterial<Fresnel, NormalGGX, Geometry> &material,
      float u1, float u2) {
    float alpha = material.roughness * material.roughness;

    float cos_theta = sqrt((1 - u1) / (1 + (alpha * alpha - 1) * u1));
    float sin_theta = sqrt(1 - cos_theta * cos_theta);
    float phi = 2 * M_PI * u2;

    return std::make_tuple(cos_theta, sin_theta, phi);
  }

  /* Sample the microfacet normal by given geometric normal, `h` is the output
   */
  template <class Fresnel, class Geometry>
  void SampleHVec(
      const MicrofacetMaterial<Fresnel, NormalGGX, Geometry> &material,
      float u1, float u2, const sycl::vec<float, 3> &n,
      sycl::vec<float, 3> &h) {
    sycl::vec<float, 3> plane_x;
    sycl::vec<float, 3> plane_y;
    /* Plane unit vectors */
    vecutils::PlaneVectors(n, plane_x, plane_y);

    sycl::vec<float, 3> _x;
    sycl::vec<float, 3> _y;
    sycl::vec<float, 3> _z;

    /* Needed for estimating in Monte Carlo integration */
    if (kUseAngleForSampling) {
      auto[theta, phi] = this->Sample(material, u1, u2);
      _x = plane_x * sin(theta) * cos(phi);
      _y = plane_y * sin(theta) * sin(phi);
      _z = n * cos(theta);

    } else {
      auto[cos_theta, sin_theta, phi] =
          this->SampleTrigonometric(material, u1, u2);
      _x = plane_x * sin_theta * cos(phi);
      _y = plane_y * sin_theta * sin(phi);
      _z = n * cos_theta;
    }
    /* Halfway vector between ingoing and outgoing directions */
    h = _x + _y + _z;
  }
};

class GeometryGGXSchlick {
  GeometryGGXSchlick(){};
  /* `n` is the halfway vector when applying the Microfacet model */
  template <class Fresnel, class Geometry>
  float operator()(
      const MicrofacetMaterial<Fresnel, NormalGGX, Geometry> &material,
      const sycl::vec<float, 3> &l, const sycl::vec<float, 3> &v,
      const sycl::vec<float, 3> &n, const sycl::vec<float, 3> &h) {
    return partial(material, l, n, h) * partial(material, v, n, h);
  }

  template <class Fresnel, class Geometry>
  float partial(
      const MicrofacetMaterial<Fresnel, NormalGGX, Geometry> &material,
      const sycl::vec<float, 3> &x, const sycl::vec<float, 3> &n,
      const sycl::vec<float, 3> &h) {
    float alpha = material.roughness * material.roughness;
    /* x * n saturated */
    float xn_dot = sycl::clamp(sycl::dot(x, n), 0.0f, 1.0f);
    /* x * h saturated */
    float xh_dot = sycl::clamp(sycl::dot(x, h), 0.0f, 1.0f);

    float tan2 = (1 - xh_dot * xh_dot) / (xh_dot * xh_dot);

    float shadow = miscutils::ShadowFactor(xh_dot / xn_dot);
    return shadow * 2 / (1 + sycl::sqrt(1 + alpha * alpha * tan2));
  }
};

} /* namespace material */

#endif