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

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda-11.3

# Install the PyTorch family (Torch, Torchvision, Torchaudio) with CUDA 11.3
# Note the use of the PyTorch extra index URL.
RUN pip install \
    torch==1.10.0+cu113 \
    torchvision==0.11.0+cu113 \
    torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html  -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install additional deps: mmcv-full, mmdet, mmsegmentation, timm
RUN pip install \
    mmcv-full==1.4.0 \
    mmdet==2.14.0 \
    mmsegmentation==0.14.1 \
    timm==0.9.5

RUN pip install opencv-python-headless av2 networkx==2.3 numpy==1.21.5 setuptools==58.2.0


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

# 安装 maptr 相关的插件
WORKDIR /opt/maptr/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
RUN python setup.py build install

# 终端显示安装信息（根据您的描述，这一步会自动完成）
# Processing dependencies for GeometricKernelAttention==1.0
# Finished processing dependencies for GeometricKernelAttention==1.0

# By default, run bash if someone logs in

CMD ["/bin/bash"]
