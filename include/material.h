#ifndef PATHTRACER_INCLUDE_MATERIAL_H_
#define PATHTRACER_INCLUDE_MATERIAL_H_

#include <cmath>
#include <sycl/sycl.hpp>

#include "include/utils.h"

/* For debugging */
const bool kUseAngleForSampling = true;

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

  SYCL_EXTERNAL MicrofacetMaterial(sycl::vec<float, 3> base_color, float metallic,
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
  SYCL_EXTERNAL void Sample(Random &random, const sycl::vec<float, 3> &v,
              const sycl::vec<float, 3> &n, sycl::vec<float, 3> &h,
              sycl::vec<float, 3> &l) {
    this->normal.SampleHVec(*this, random(), random(), n, h);
    /* Sampled outgoing direction (light direction) */
    l = h;
  }

  /* Returns the summation element for the monte carlo estimator */
  SYCL_EXTERNAL float Eval(const sycl::vec<float, 3> &l, const sycl::vec<float, 3> &v,
             const sycl::vec<float, 3> &n, const sycl::vec<float, 3> &h) {
    /* Fresnel term in BRDF */
    sycl::vec<float, 3> F = this->fresnel(*this, h, l);
    /* Geometric term in BRDF */
    float G = this->geometry(*this, l, v, n, h);

    if (sycl::dot(n,l) > 0.0f && sycl::dot(l, h) > 0.0f) {
      /* Denominator term */
      float denominator = sycl::dot(n, v)*
                          sycl::dot(n, h);

      return G * sycl::fabs(sycl::dot(v, h)) / denominator;
    }
    return 0.0f;
  }
};

/* Function objects */
class FresnelSchlick {
public:
  SYCL_EXTERNAL FresnelSchlick(){};

  /* One fresnel for both dielectric and metallic materials */
  /* `n` is the halfway vector when applying the Microfacet model */
  template <class Fresnel, class Normal, class Geometry>
  SYCL_EXTERNAL sycl::vec<float, 3> operator()(
      const MicrofacetMaterial<Fresnel, Normal, Geometry> &material,
      const sycl::vec<float, 3> &n, const sycl::vec<float, 3> &v) {
    sycl::vec<float, 3> F0{0.0f, 0.0f,
                           0.0f};

    /* Linear mix f0 and base color based on metaliness */
    F0 = vecutils::Lerp(F0, material.base_color, material.metallic);

    sycl::vec<float, 3> ones{1.0f, 1.0f, 1.0f};
    float rdot = 1.0f - sycl::clamp(sycl::dot(n, v), 0.0f, 1.0f);

    return F0 + (ones - F0) * (rdot * rdot * rdot * rdot * rdot);
  }
};

class NormalGGX {
public:
  SYCL_EXTERNAL NormalGGX(){};
  /* `n` is the halfway vector when applying the Microfacet model */
  template <class Fresnel, class Geometry>
  SYCL_EXTERNAL float operator()(
      const MicrofacetMaterial<Fresnel, material::NormalGGX, Geometry> &material,
      const sycl::vec<float, 3> &n, const sycl::vec<float, 3> &h) {
    float alpha = material.roughness * material.roughness;
    float alpha_sq = alpha * alpha;
    float dot = sycl::clamp(sycl::dot(n, h), 0.0f, 1.0f);

    float tmp = dot * dot * alpha_sq + (1 - dot * dot);
    return (miscutils::ShadowFactor(dot) * alpha_sq) / (M_PI * tmp * tmp);
  }

  /* Returns whole angles for theta and phi */
  template <class Fresnel, class Geometry>
  SYCL_EXTERNAL std::pair<float, float> Sample(
      const MicrofacetMaterial<Fresnel, material::NormalGGX, Geometry> &material,
      float u1, float u2) {
    float alpha = material.roughness * material.roughness;

    /* Sampling of normal distribution function */

    float phi = u1 * M_PI_2;
    float theta = sycl::atan(alpha*sycl::sqrt(u1 / (1.0f - u1)));
    theta = u2 * 2 * M_PI;

    return std::pair<float, float>(phi, theta);
  }

  /* Returns theta in cos and sin, while phi is returned as an angle. It is more
   * efficient to compute this */
  template <class Fresnel, class Geometry>
  SYCL_EXTERNAL std::tuple<float, float, float> SampleTrigonometric(
      const MicrofacetMaterial<Fresnel, NormalGGX, Geometry> &material,
      float u1, float u2) {
    float alpha = material.roughness * material.roughness;

    float cos_theta = sycl::sqrt((1 - u1) / (1 + (alpha * alpha - 1) * u1));
    float sin_theta = sycl::sqrt(1 - cos_theta * cos_theta);
    float phi = 2 * M_PI * u2;

    return std::make_tuple(cos_theta, sin_theta, phi);
  }

  /* Sample the microfacet normal by given geometric normal, `h` is the output
   */
  template <class Fresnel, class Geometry>
  SYCL_EXTERNAL void SampleHVec(
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
      auto[phi, theta] = this->Sample(material, u1, u2);
      _x = plane_x * sycl::sin(phi) * sycl::cos(theta);
      _y = plane_y * sycl::sin(phi) * sycl::sin(theta);
      _z = n * sycl::cos(phi);

    } else {
      auto[cos_theta, sin_theta, phi] =
          this->SampleTrigonometric(material, u1, u2);
      _x = plane_x * sin_theta * sycl::cos(phi);
      _y = plane_y * sin_theta * sycl::sin(phi);
      _z = n * cos_theta;
    }
    /* Halfway vector between ingoing and outgoing directions */
    h = _x + _y + _z;
  }
};

class GeometryGGXSchlick {
public:
  SYCL_EXTERNAL GeometryGGXSchlick(){};
  /* `n` is the halfway vector when applying the Microfacet model */
  template <class Fresnel, class Geometry>
  SYCL_EXTERNAL float operator()(
      const MicrofacetMaterial<Fresnel, NormalGGX, Geometry> &material,
      const sycl::vec<float, 3> &l, const sycl::vec<float, 3> &v,
      const sycl::vec<float, 3> &n, const sycl::vec<float, 3> &h) {
    return partial(material, l, n, h) * partial(material, v, n, h);
  }

  template <class Fresnel, class Geometry>
  SYCL_EXTERNAL float partial(
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