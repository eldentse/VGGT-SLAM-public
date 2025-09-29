# Zapvision Linux Build And Demo

- =======================================
- Jordan Campbell (R&D Software Engineer)
- jordan@zappar.com
- 21 May 2025
- Version 1.3
- =======================================

This directory contains the header file `Zapvision.h` and libraries in `libs`.

At the moment the libraries are:

- libZapvision.a
- libopencv_calib3d.a
- libopencv_core.a
- libopencv_features2d.a
- libopencv_flann.a
- libopencv_imgproc.a
- libopencv_video.a
- libz.a

Ideally we would only distribute a single `libZapvision.a` that contained all the opencv libs packaged nicely, however linking against these wasn't playing nicely so in the interest of time I'm just distributing them separately. (So you'll need them all for now). The CMakeLists.txt file shows how to link against these:

```
target_link_libraries(
    zapvision
    PUBLIC
    ${CMAKE_SOURCE_DIR}/libs/libZapvision.a
    ${CMAKE_SOURCE_DIR}/libs/libopencv_video.a
    ${CMAKE_SOURCE_DIR}/libs/libopencv_calib3d.a
    ${CMAKE_SOURCE_DIR}/libs/libopencv_flann.a
    ${CMAKE_SOURCE_DIR}/libs/libopencv_features2d.a
    ${CMAKE_SOURCE_DIR}/libs/libopencv_imgproc.a
    ${CMAKE_SOURCE_DIR}/libs/libopencv_core.a
    ${CMAKE_SOURCE_DIR}/libs/libz.a
)
```

Of course you're free to compile everything however you wish, but I can at least confirm that this setup works in this test repo.

If you don't want frame to frame tracking (just raw detections per frame) then can avoid needing to link against opencv entirely.

### Quickstart

Running `./test_zapvision` should produce the following output:

```
root@2cc529bec9db:/app/ZapvisionLinux# ./test_zapvision
Loading spec data ZAPVISION_2_2_4_V2_ZBS
Loading spec data ZAPVISION_2_2_4_V3_ZBS
Loading spec data ZAPVISION_2_2_4_V1_ZBS
Loading spec data ZAPVISION_DENSE_ZBS
Zapvision 3.2.11
read 8294400 bytes [3840, 2160]
Have 24 markers.
aqr: category 8503, qr [], nlandmarks 3
aqr: category 10604, qr [], nlandmarks 3
aqr: category 14313, qr [], nlandmarks 3
aqr: category 3760, qr [], nlandmarks 3
aqr: category 14133, qr [], nlandmarks 3
aqr: category 2478, qr [], nlandmarks 3
aqr: category 7296, qr [], nlandmarks 3
aqr: category 1716, qr [], nlandmarks 3
aqr: category 5267, qr [], nlandmarks 3
aqr: category 3161, qr [], nlandmarks 3
aqr: category 595, qr [], nlandmarks 3
aqr: category 3871, qr [], nlandmarks 3
aqr: category 5983, qr [], nlandmarks 3
aqr: category 1951, qr [], nlandmarks 3
aqr: category 2073, qr [], nlandmarks 3
dense: 8112, nlandmarks 4
dense: 15121, nlandmarks 4
dense: 12008, nlandmarks 4
dense: 12275, nlandmarks 4
dense: 1450, nlandmarks 4
aqr: category 1069, qr [], nlandmarks 3
dense: 7700, nlandmarks 4
dense: 11226, nlandmarks 4
dense: 2752, nlandmarks 4
```

### Build

To build this demo project you can just run:

```
cd ZapvisionLinux
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/zapvision
```

If everything works as expected then you should get the same output as above (that demo build is just a precompiled binary of this repo).

The `main.cpp` file loads the image in `image.data`. This is just binary image data in the format: `<width><height><...data...>` (i.e it's uncompressed bytes, which is why it's a large file). I've included the file in this format as ultimately it's what the zapvision processor expects. You can of course provide images to the application in any format, so long as `zapvision_tracker_process( ... )` receives _a pointer to a block of memory_. I've also included a screenshot of this particular image for reference. The `Image` struct in this example is just for convenience and doesn't have to be used.

Note that Image.png is just a screenshot of the actual image, so shouldn't be used for anything. You can export `image.data` into a different format if you'd like to test it elsewhere.

## Pybind

I've included some demo code: `bindings.cpp`, `zapvision.py` and and update to `CMakeLists.txt` in order to confirm and test that the `-fPIC` flag has successfully propagated through everywhere. This just runs on dummy data so isn't expected to log out any valid input.

Enjoy! Any problems or questions please let me know.

## Changelog

- 2025.05.22 [1.1] Explicitly build for x64-64 rather than aarch64 (updated dockerfile).
- 2025.05.23 [1.3] Add `-fPIC` flag to libraries, add pybind test and confirm working
