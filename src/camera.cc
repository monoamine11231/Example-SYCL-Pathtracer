#include "include/camera.h"

#include <cmath>

void Camera::GenerateWorkVariables() {
  sycl::vec<float, 3> image_center;
  float aspect_ratio, fov_tan, image_width, image_height;

  aspect_ratio = (float)this->pheight_ / (float)this->pwidth_;

  fov_tan = std::tan((this->fov_ / 360.0f) * M_PI);

  image_width = fov_tan * this->focal_length_ * 2;
  image_height = image_width * aspect_ratio;

  this->w_factor_ = image_width / this->pwidth_;
  this->h_factor_ = image_height / this->pheight_;

  image_center = this->dir_ * this->focal_length_;
  this->image_corner_ = image_center - this->right_ * image_width / 2 +
                        this->up_ * image_height / 2;
}

/* Main idea is that camera dir can be updated more often that fov or fb
 * dimensions */
void Camera::LookAt(sycl::vec<float, 3> dir, const sycl::vec<float, 3>& up) {
  this->dir_ = dir;

  this->right_ = sycl::normalize(sycl::cross(this->dir_, up));
  this->up_ = -sycl::normalize(sycl::cross(this->dir_, this->right_));

  this->GenerateWorkVariables();
}

Camera::Camera(sycl::vec<float, 3> dir, sycl::vec<float, 3> origin,
               const sycl::vec<float, 3>& up, float fov, float focal_length,
               uint16_t pwidth, uint16_t pheight) {
  this->dir_ = dir;
  this->origin_ = origin;

  this->fov_ = fov;
  this->focal_length_ = focal_length;

  this->pwidth_ = pwidth;
  this->pheight_ = pheight;

  /* Generate the remaining tmp variables for later ray generation */
  this->LookAt(dir, up);
}

void Camera::GenerateRay(int w, int h, Ray& ray) const {
  /* OpenGL buffers start from bottom left corner */
  h = this->pheight_ - h;

  sycl::vec<float, 3> dir = this->image_corner_ +
                            this->right_ * this->w_factor_ * w -
                            this->up_ * this->h_factor_ * h;

  ray.dir = sycl::normalize(dir);
  ray.origin = this->origin_;
}

void Camera::UpdateFOV(float fov) {
  this->fov_ = fov;

  this->GenerateWorkVariables();
}

void Camera::UpdateFocalLength(float focal_length) {
  this->focal_length_ = focal_length;

  this->GenerateWorkVariables();
}

void Camera::UpdateDimensions(uint16_t pwidth, uint16_t pheight) {
  this->pwidth_ = pwidth;
  this->pheight_ = pheight;

  this->GenerateWorkVariables();
}

sycl::vec<float, 3> Camera::GetFront() const {
  return this->dir_;
}

sycl::vec<float, 3> Camera::GetRight() const {
  return this->right_;
}

sycl::vec<float, 3> Camera::GetUp() const {
  return this->up_;
}