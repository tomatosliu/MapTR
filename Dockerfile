# Start from NVIDIA's CUDA 11.3 base image (with cudnn8, if desired)
# This ensures CUDA 11.3 is already installed
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8, pip and required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    curl python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Make Python 3.8 your default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1


# Set proxy variables for Oasa env
ENV HTTP_PROXY=http://192.168.1.181:7890
ENV HTTPS_PROXY=http://192.168.1.181:7890
ENV EXPORT_ALL_PROXY=socks5://192.168.1.181:7890

# Install pip for Python 3.8
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py


# Install the PyTorch family (Torch, Torchvision, Torchaudio) with CUDA 11.3
# Note the use of the PyTorch extra index URL.
RUN pip install \
    torch==1.10.0+cu113 \
    torchvision==0.11.0+cu113 \
    torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install additional deps: mmcv-full, mmdet, mmsegmentation, timm
RUN pip install \
    mmdet==2.14.0 \
    mmsegmentation==0.14.1 \
    timm==0.9.5 \
    opencv-python-headless av2 networkx==2.3 numpy==1.21.5 setuptools==58.2.0
RUN pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda-11.3
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CPATH=${CUDA_HOME}/include:${CPATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.6;7.5;7.0"

COPY . /opt/maptr
WORKDIR /opt/maptr/mmdetection3d
RUN pip install -r requirements.txt
RUN pip install -v -e .
# RUN rm -rf /opt/maptr

RUN pip install numpy==1.21.5 yapf==0.31.0
CMD ["/bin/bash"]

# docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --network=host -v /media/ftp:/media/ftp -itd --name maptr_nv maptr
# docker exec -it maptr_nv /bin/bash
# 安装 maptr 相关的插件
# cd /opt/maptr/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
# python setup.py build install
# cd /opt/maptr/
# PYTHONPATH=$(pwd) python tools/maptr/vis_pred.py projects/configs/maptr/maptr_tiny_r50_24e.py ckpts/maptr_tiny_r50_24e.pth --show-dir ./vis_dir