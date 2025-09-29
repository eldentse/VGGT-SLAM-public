#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "Zapvision.h"

namespace py = pybind11;

class ZapvisionTracker {
public:
    ZapvisionTracker() {
        tracker_ = zapvision_tracker_create();
    }

    ~ZapvisionTracker() {
        zapvision_tracker_destroy(tracker_);
    }

    void process(py::array_t<unsigned char> image, int width, int height, int stride, float fov = 0.0f) {
        py::buffer_info buf = image.request();
        unsigned char* data_ptr = static_cast<unsigned char*>(buf.ptr);
        zapvision_tracker_process(tracker_, data_ptr, width, height, stride, fov);
    }

    int result_count() const {
        return zapvision_tracker_result_count(tracker_);
    }

    py::array_t<float> result_pose(int index) const {
        float* ptr = zapvision_tracker_result_pose(tracker_, index);
        return py::array_t<float>({6}, ptr);
    }

    int result_product_category(int index) const {
        return zapvision_tracker_result_product_category(tracker_, index);
    }

    std::string result_qr_code(int index) const {
        const char* code = zapvision_tracker_result_qr_code(tracker_, index);
        return code ? std::string(code) : std::string("");
    }

    int result_type(int index) const {
        return zapvision_tracker_result_type(tracker_, index);
    }

    unsigned result_dense_code_value(int index) const {
        return zapvision_tracker_result_dense_code_value(tracker_, index);
    }

    py::array_t<float> projection_matrix(int render_width, int render_height, float z_near, float z_far, int orientation) const {
        float* ptr = zapvision_tracker_projection_matrix(tracker_, render_width, render_height, z_near, z_far, static_cast<screen_orientation>(orientation));
        return py::array_t<float>({16}, ptr);
    }

    py::array_t<unsigned char> perform_raw_data_copy(py::buffer raw_data, int width, int height, int stride) {
        py::buffer_info buf = raw_data.request();
        auto* result = zapvision_tracker_perform_raw_data_copy(tracker_, buf.ptr, width, height, stride);
        return py::array_t<unsigned char>({height, stride}, result);
    }

    py::array_t<unsigned char> convert_rgb_to_luminance(py::buffer r, py::buffer g, py::buffer b, int width, int height, int stride) {
        auto* result = zapvision_tracker_convert_rgb_to_luminance(tracker_, r.request().ptr, g.request().ptr, b.request().ptr, width, height, stride);
        return py::array_t<unsigned char>({height, stride}, result);
    }

    py::array_t<unsigned char> convert_rgb_single_plane_to_luminance(py::buffer raw_data, int width, int height, int stride, int layout) {
        auto* result = zapvision_tracker_convert_rgb_single_plane_to_luminance(tracker_, raw_data.request().ptr, width, height, stride, static_cast<pixel_layout>(layout));
        return py::array_t<unsigned char>({height, stride}, result);
    }

    py::array_t<float> result_landmarks(int index) const {
        float* ptr = zapvision_tracker_result_landmarks(tracker_, index);
        int count = zapvision_tracker_result_landmarks_count(tracker_, index);
        return py::array_t<float>({count * 2}, ptr);
    }

    py::array_t<float> result_qr_relative_pose(int index) const {
        float* ptr = zapvision_tracker_result_qr_relative_pose(tracker_, index);
        return py::array_t<float>({6}, ptr);
    }

private:
    zapvision_tracker tracker_;
};

PYBIND11_MODULE(_zapvision_py, m) {
    py::enum_<screen_orientation>(m, "ScreenOrientation")
        .value("UNKNOWN", unknown)
        .value("PORTRAIT", portrait)
        .value("LANDSCAPE_LEFT", landscape_left)
        .value("PORTRAIT_UPSIDE_DOWN", portrait_upside_down);

    py::enum_<pixel_layout>(m, "PixelLayout")
        .value("RGBA", rgba)
        .value("ARGB", argb)
        .value("RGB", rgb);

    py::class_<ZapvisionTracker>(m, "ZapvisionTracker")
        .def(py::init<>())
        .def("process", &ZapvisionTracker::process, py::arg("image"), py::arg("width"), py::arg("height"), py::arg("stride"), py::arg("fov") = 0.0f)
        .def("result_count", &ZapvisionTracker::result_count)
        .def("result_pose", &ZapvisionTracker::result_pose)
        .def("result_product_category", &ZapvisionTracker::result_product_category)
        .def("result_qr_code", &ZapvisionTracker::result_qr_code)
        .def("result_type", &ZapvisionTracker::result_type)
        .def("result_dense_code_value", &ZapvisionTracker::result_dense_code_value)
        .def("projection_matrix", &ZapvisionTracker::projection_matrix)
        .def("perform_raw_data_copy", &ZapvisionTracker::perform_raw_data_copy)
        .def("convert_rgb_to_luminance", &ZapvisionTracker::convert_rgb_to_luminance)
        .def("convert_rgb_single_plane_to_luminance", &ZapvisionTracker::convert_rgb_single_plane_to_luminance)
        .def("result_landmarks", &ZapvisionTracker::result_landmarks)
        .def("result_qr_relative_pose", &ZapvisionTracker::result_qr_relative_pose);
}
