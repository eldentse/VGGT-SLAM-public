FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

RUN apt-get update && \
    apt-get install -y libgl1 libomp5 libunwind-14t64 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install open3d scikit-build cmake ninja --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130

COPY . /workspaces/VGGT-SLAM-public

RUN cd /workspaces/VGGT-SLAM-public/ZapvisionLinux && \
    sed -i '/#include <iostream>/a #include <cstdint>' main.cpp && \
    pip3 install .

RUN cd /workspaces/VGGT-SLAM-public && \
    ./setup.sh

WORKDIR /workspaces/VGGT-SLAM-public