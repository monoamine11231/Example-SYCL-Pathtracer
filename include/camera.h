#ifndef PATHTRACER_INCLUDE_CAMERA_H_
#define PATHTRACER_INCLUDE_CAMERA_H_

#include <sycl/sycl.hpp>


/*  The main philosophy here is that everything that is not private can be changed
    without a setter */
class Camera {
private:
    sycl::vec<float, 3>     dir_;           /* Modifying this changes all the tmp vars */

    /* Tmp vars generated by `lookat` method for generating rays later on */
    sycl::vec<float, 3>     up_;            /* Up based on camera view direction */
    sycl::vec<float, 3>     right_;         /* Right based on camera view direction */

    sycl::vec<float, 3>     image_corner_;  /* Image left top corner 4 ray generation */

    float                   w_factor_;      /* Horizontal shifting factor 4 every pixel*/
    float                   h_factor_;      /* Vertical shifting factor 4 every pixel*/

public:
    sycl::vec<float, 3>     origin_;

    float                   fov_;           /* In degrees */
    float                   focal_length_;


    Camera(sycl::vec<float, 3> dir, sycl::vec<float, 3> origin,
           float fov, float focal_length,
           uint16_t pwidth, uint16_t pheight); /* 65536 x 65536 max resolution for fb */

    void lookat(sycl::vec<float, 3> dir);
};

#endif