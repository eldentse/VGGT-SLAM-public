#ifndef ZAPVISION_H_
#define ZAPVISION_H_

#ifdef __cplusplus
extern "C"
{
#endif

  enum screen_orientation
  {
    unknown = 0,
    portrait,
    landscape_left,
    portrait_upside_down,
  };

  enum pixel_layout
  {
    rgba = 0,
    argb,
    rgb
  };

  //  The number of 2D landmarks returned by zapvision_tracker_result_landmarks, below.

  typedef struct zapvision_tracker_t *zapvision_tracker;

  zapvision_tracker zapvision_tracker_create();
  void zapvision_tracker_destroy(zapvision_tracker tracker);

  /**
   *
   * It is the responsibility of the caller to ensure that image memory is valid throughout the lifetime
   * of the zapvision_tracker_process() function.
   *
   */
  void zapvision_tracker_process(zapvision_tracker tracker, unsigned char *data, const int width, const int height, const int stride, const float fov = 0.0f);

  /**
   *
   * The following functions will return results that correspond to the immediately preceeding
   * call to zapvision_tracker_process().
   *
   * Subsequent calls to zapvision_tracker_process() may invalidate memory returned by these
   * functions.
   *
   */
  int zapvision_tracker_result_count(zapvision_tracker tracker);
  float *zapvision_tracker_result_pose(zapvision_tracker tracker, int index);
  int zapvision_tracker_result_product_category(zapvision_tracker tracker, int index);
  const char *zapvision_tracker_result_qr_code(zapvision_tracker tracker, int index);

  // -1: invalid
  //  0: aqr (category + (optional)qr)
  //  1: dense code
  int zapvision_tracker_result_type(zapvision_tracker handle, int index);

  unsigned zapvision_tracker_result_dense_code_value(zapvision_tracker handle, int index);

  float *zapvision_tracker_projection_matrix(zapvision_tracker tracker, int render_width, int render_height, float z_near, float z_far, enum screen_orientation orientation);

  unsigned char *zapvision_tracker_perform_raw_data_copy(zapvision_tracker handle, const void *raw_data, const int width, const int height, const int stride);
  unsigned char *zapvision_tracker_convert_rgb_to_luminance(zapvision_tracker handle, const void *r, const void *g, const void *b, const int width, const int height, const int stride);
  unsigned char *zapvision_tracker_convert_rgb_single_plane_to_luminance(zapvision_tracker handle, const void *raw_data, const int width, const int height, const int stride, const pixel_layout layout);


  float *zapvision_tracker_result_landmarks(zapvision_tracker handle, int index);
  int zapvision_tracker_result_landmarks_count(zapvision_tracker handle, int index);
  float *zapvision_tracker_result_qr_relative_pose(zapvision_tracker handle, int index);

#ifdef __cplusplus
}
#endif

#endif // ZAPVISION_H_
