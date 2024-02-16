#include "include/objects/mesh.h"

#include <sycl/sycl.hpp>

/* Ray-triangle intersection */
std::optional<Intersector> MeshTriangle::Intersect(const Ray& ray) const {
  std::optional<Intersector> intersection;
  sycl::vec<float, 3> P, C, edge, vp;
  float ndotdir, d, t;

  ndotdir = sycl::dot(this->normal_, ray.dir);
  if (ndotdir == 0.0f) {
    /* Empty intersection, no value */
    return intersection;
  }

  d = -sycl::dot(this->normal_, this->a_);
  t = -(sycl::dot(this->normal_, ray.origin) + d) / ndotdir;

  if (t < 0.0f) {
    /* Behind triangle return empty intersection */
    return intersection;
  }

  P = ray.origin + t * ray.dir;

  /* First edge check if outside or inside triangle */
  edge = this->b_ - this->a_;
  vp = P - this->a_;
  C = sycl::cross(edge, vp);
  if (sycl::dot(this->normal_, C) < 0.0f) {
    /* On right side of triangle, return empty intersection */
    return intersection;
  }

  edge = this->c_ - this->b_;
  vp = P - this->b_;
  C = sycl::cross(edge, vp);
  if (sycl::dot(this->normal_, C) < 0.0f) {
    /* On left side of triangle, return empty intersection */
    return intersection;
  }

  edge = this->a_ - this->c_;
  vp = P - this->c_;
  C = sycl::cross(edge, vp);
  if (sycl::dot(this->normal_, C) < 0.0f) {
    /* On right side of triangle, return empty intersection */
    return intersection;
  }

  /* If inside data, initialize the struct with the intersection and face data
   */
  Intersector data(t, this->normal_, this->material_id_);
  intersection = data;

  return intersection;
}

Mesh::Mesh(const rapidobj::Shape& shape,
           const rapidobj::Attributes& attributes) {
  for (std::size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
    int pindex1 = shape.mesh.indices[i + 0].position_index;
    int pindex2 = shape.mesh.indices[i + 1].position_index;
    int pindex3 = shape.mesh.indices[i + 2].position_index;

    sycl::vec<float, 3> a, b, c, normal;
    a = sycl::vec<float, 3>(attributes.positions[pindex1 + 0],
                            attributes.positions[pindex1 + 1],
                            attributes.positions[pindex1 + 2]);

    b = sycl::vec<float, 3>(attributes.positions[pindex2 + 0],
                            attributes.positions[pindex2 + 1],
                            attributes.positions[pindex2 + 2]);

    c = sycl::vec<float, 3>(attributes.positions[pindex3 + 0],
                            attributes.positions[pindex3 + 1],
                            attributes.positions[pindex3 + 2]);

    normal = sycl::cross(b - a, c - a);
    normal = sycl::normalize(normal);

    this->faces_.push_back(
        MeshTriangle(a, b, c, normal, shape.mesh.material_ids[i / 3]));
  }
}

/*  Loop through every triangular face in the mesh and find the closest
    intersection if there is any. */
std::optional<Intersector> Mesh::Intersect(const Ray& ray) const {
  std::optional<Intersector> intersection;
  for (const auto& face : this->faces_) {
    std::optional<Intersector> new_intersection = face.Intersect(ray);
    if (!new_intersection.has_value()) {
      continue;
    }

    if (!intersection.has_value()) {
      intersection = new_intersection;
      continue;
    }

    if (new_intersection->t < intersection->t) {
      intersection = new_intersection;
    }
  }

  return intersection;
}