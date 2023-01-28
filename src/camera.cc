#include <cmath>
#include "include/camera.h"


void Camera::GenerateWorkVariables() {
    sycl::vec<float, 3> forward, image_center;
    float aspect_ratio, fov_tan, image_width, image_height;


    aspect_ratio = (float)this->pheight_/(float)this->pwidth_;

    forward = this->dir_*-1.0f;
    
    fov_tan = std::tan((this->fov_ / 360.0f) * M_PI);
    
    image_width = fov_tan * this->focal_length_ * 2;
    image_height = image_width * aspect_ratio;

    this->w_factor_ = image_width / this->pwidth_;
    this->h_factor_ = image_height / this->pheight_;

    image_center = this->dir_*this->focal_length_;
    this->image_corner_=image_center-this->right_*image_width/2+this->up_*image_height/2;
}

/* Main idea is that camera dir can be updated more often that fov or fb dimensions */
void Camera::LookAt(sycl::vec<float, 3> dir, const sycl::vec<float, 3>& up) {
    this->dir_      = dir;

    this->right_    = sycl::cross(up, this->dir_*-1.0f);
    this->up_       = sycl::cross(this->dir_*-1.0f, this->right_);

    this->GenerateWorkVariables();
}

Camera::Camera(sycl::vec<float, 3> dir, sycl::vec<float, 3> origin,
               const sycl::vec<float, 3>& up,
               float fov, float focal_length,
               uint16_t pwidth, uint16_t pheight) {

    this->dir_          = dir;
    this->origin_       = origin;

    this->fov_          = fov;
    this->focal_length_ = focal_length;

    this->pwidth_       = pwidth;
    this->pheight_      = pheight;

    /* Generate the remaining tmp variables for later ray generation */
    this->LookAt(dir, up);
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