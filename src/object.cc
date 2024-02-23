#include "include/object.h"

/* Returns the closest intersection for the ray in the vector of given objects,
 * if exists */
std::optional<Intersector> closest_obj(
    const Ray &ray, const containerutils::VariantContainer<Objects> &objects) {
  std::optional<Intersector> global_intersection{};
  /* Lambda that checks and overwrites `global_intersection` if the given object
   * has a closer intersection than the previous one */
  auto evaluator = [&ray, &global_intersection](const auto &obj) {
    std::optional<Intersector> intersection = obj.Intersect(ray);
    if (!intersection.has_value())
      return;

    if (!global_intersection.has_value()) {
      global_intersection = intersection;
      return;
    }

    /* Overwrite if a closer intersection was found */
    if (global_intersection->t > intersection->t) {
      global_intersection = intersection;
    }
  };

  objects.forEach(evaluator);
  return global_intersection;
}