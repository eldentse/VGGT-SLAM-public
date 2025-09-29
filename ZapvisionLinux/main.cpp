#include "Zapvision.h"

#include <fstream>
#include <iostream>

struct Image {
  int width = 0;
  int height = 0;
  uint8_t *data = nullptr;

  Image(const std::string &filepath) {
    std::ifstream file{filepath, std::ios::binary};

    file.read(reinterpret_cast<char *>(&width), sizeof(int));
    file.read(reinterpret_cast<char *>(&height), sizeof(int));

    const int nbytes = width * height;

    data = new uint8_t[nbytes];
    if (!data) {
      throw std::runtime_error("Error: Could not allocate memory for Image.");
    }

    if (file.eof()) {
      delete[] data;
      throw std::runtime_error("Error: No image data in file.");
    }

    file.read(reinterpret_cast<char *>(data), nbytes * sizeof(uint8_t));

    std::cout << "read " << nbytes << " bytes [" << width << ", " << height
              << "]" << std::endl;
  }

  ~Image() {
    delete[] data;
    data = nullptr;
  }
};

int main() {

  zapvision_tracker z = zapvision_tracker_create();

  try {

    Image img{"./image.data"};

    // The last parameter is the stride, which is the number of bytes across a single row.
    // For most cases the stride will be the same as the width, however there are two cases
    //  where this differs: (1) If the data is in RGB format for example then the stride is
    //  3 * width, and (2) if there are buffer pixels at the end of each row, in which case
    //  the stride is the width + number of buffer bytes. Note that we don't support case (1)
    //  but case (2) should work fine if this is the image format your application provides.
    zapvision_tracker_process(z, img.data, img.width, img.height, img.width);

    const int n = zapvision_tracker_result_count(z);

    std::cout << "Have " << n << " markers." << std::endl;

    for (int i = 0; i < n; ++i) {

      // -1: invalid
      //  0: aqr (category + (optional)qr)
      //  1: dense code
      const int marker_type = zapvision_tracker_result_type(z, i);

      if (marker_type < 0) {
        std::cout << "Invalid marker." << std::endl;
        continue;
      }

      if (marker_type == 0) {
        const int category = zapvision_tracker_result_product_category(z, i);
        const std::string qr = zapvision_tracker_result_qr_code(z, i);
        const int nlandmarks = zapvision_tracker_result_landmarks_count(z, i);
        const float *landmarks = zapvision_tracker_result_landmarks(z, i);

        std::cout << "aqr: category " << category << ", qr [" << qr
                  << "], nlandmarks " << nlandmarks << std::endl;

      } else {
        // dense code
        const int value = zapvision_tracker_result_dense_code_value(z, i);
        const int nlandmarks = zapvision_tracker_result_landmarks_count(z, i);
        std::cout << "dense: " << value << ", nlandmarks " << nlandmarks
                  << std::endl;
      }
    }

  } catch (std::exception &error) {
    std::cout << error.what() << std::endl;
    zapvision_tracker_destroy(z);
    return 1;
  }

  zapvision_tracker_destroy(z);

  return 0;
}
