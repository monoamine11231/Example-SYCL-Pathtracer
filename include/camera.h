#ifndef PATHTRACER_INCLUDE_CAMERA_H_
#define PATHTRACER_INCLUDE_CAMERA_H_

#include <sycl/sycl.hpp>

#include "include/ray.h"

/*  The main philosophy here is that everything that is not private can be
   changed without a setter */
class Camera {
 private:
  void GenerateWorkVariables();

  sycl::vec<float, 3> dir_; /* Modifying this changes all the tmp vars */

  float fov_; /* In degrees */
  float focal_length_;

  uint16_t pwidth_;  /* Image width in pixels */
  uint16_t pheight_; /* Image height in pixels */

  /* Tmp vars generated by `lookat` method for generating rays later on */
  sycl::vec<float, 3> up_;    /* Up based on camera view direction */
  sycl::vec<float, 3> right_; /* Right based on camera view direction */

  sycl::vec<float, 3>
      image_corner_; /* Image left top corner for ray generation */

  float w_factor_; /* Horizontal shifting factor for every pixel*/
  float h_factor_; /* Vertical shifting factor for every pixel*/

 public:
  sycl::vec<float, 3> origin_;

  Camera(sycl::vec<float, 3> dir, sycl::vec<float, 3> origin,
         const sycl::vec<float, 3>& up, float fov, float focal_length,
         uint16_t pwidth,
         uint16_t pheight); /* 65536 x 65536 max resolution for fb */

  /* Generates the ray from the given pixel on the image and local work camera
   * variables */ 
  SYCL_EXTERNAL void GenerateRay(uint16_t w, uint16_t h, Ray& ray) const;

  void LookAt(sycl::vec<float, 3> dir, const sycl::vec<float, 3>& up);

  void UpdateFOV(float fov);
  void UpdateFocalLength(float focal_length);
  void UpdateDimensions(uint16_t pwidth, uint16_t pheight);
};

#endif