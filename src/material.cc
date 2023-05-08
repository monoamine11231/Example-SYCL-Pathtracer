#include "include/material.h"

#include <cmath>

float material::FresnelSchlick::dielectric(float f0,
                                           const sycl::vec<float, 3> &n,
                                           const sycl::vec<float, 3> &v) {
  float dot = sycl::dot(n, v);
  /* Faster than pow */
  return f0 + (1 - f0) * dot * dot * dot * dot * dot;
}

float material::NormalGGX::operator()(float roughness,
                                      const sycl::vec<float, 3> &n,
                                      const sycl::vec<float, 3> &h) {
  float alpha = roughness * roughness;
  float dot = sycl::dot(n, h);

  float intermediate = dot * dot * (alpha * alpha - 1) + 1;
  return (alpha * alpha) / (M_PI * intermediate * intermediate);
}

float material::GeometryGGXSchlick::operator()(float roughness,
                                               const sycl::vec<float, 3> &n,
                                               const sycl::vec<float, 3> &v) {
  float k = roughness * roughness * 0.5;
  float dot = sycl::dot(n, v);
  return dot / (dot * (1 - k) + k);
}