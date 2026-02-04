# syntax=docker/dockerfile:1.4

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION

FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION} AS simulation

ARG ISAACSIM_ROOT_PATH=/isaac-sim
ARG ISAACLAB_PATH=/workspace/isaaclab
ARG DOCKER_USER_HOME=/root
ARG ROS2_DISTRO=jazzy
ARG ROS2_APT_PACKAGE=ros-base
ARG ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git
ARG ISAACLAB_REF

ENV ISAACSIM_VERSION=${ISAACSIM_VERSION} \
    ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH} \
    ISAACLAB_PATH=${ISAACLAB_PATH} \
    DOCKER_USER_HOME=${DOCKER_USER_HOME} \
    ROS_DISTRO=${ROS2_DISTRO} \
    ROS2_APT_PACKAGE=${ROS2_APT_PACKAGE} \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    # RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    # FASTRTPS_DEFAULT_PROFILES_FILE=${DOCKER_USER_HOME}/.ros/fastdds.xml \
    # CYCLONEDDS_URI=${DOCKER_USER_HOME}/.ros/cyclonedds.xml \
    ISAACSIM_PATH=${ISAACLAB_PATH}/_isaac_sim \
    OMNI_KIT_ALLOW_ROOT=1 \
    http_proxy=http://127.0.0.1:8889 \
    https_proxy=http://127.0.0.1:8889 \
    HTTP_PROXY=http://127.0.0.1:8889 \
    HTTPS_PROXY=http://127.0.0.1:8889 \
    no_proxy=localhost,127.0.0.1,::1 \
    NO_PROXY=localhost,127.0.0.1,::1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

SHELL ["/bin/bash", "-c"]

USER root

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
    build-essential \
    cmake \
    git \
    git-lfs \
    libglib2.0-0 \
    ncurses-term \
    wget \
    curl \
    gedit \
    tmux \
    htop \
    software-properties-common \
    vim \
    libgtk2.0-dev \
    libgtk-3-dev \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-util1 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    pkg-config \
    libfontconfig1-dev \
    libfreetype6-dev \
    libfreetype-dev \
    # libbrotli1=1.0.9-2build6 \
    # libbrotli-dev=1.0.9-2build6 \
    libharfbuzz-dev \
    libpango1.0-dev \
    libxft-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module

RUN git clone --branch ${ISAACLAB_REF} --depth 1 ${ISAACLAB_REPO} ${ISAACLAB_PATH} && \
    rm -rf ${ISAACLAB_PATH}/.git

RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh

RUN ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --upgrade pip && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml SciencePlots pytest lark usd-core cvxpy onnx && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install moviepy==1.0.3 addict plyfile pycolmap evo && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install e3nn manifold3d onnxruntime-gpu

RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source

RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/ov && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/pip && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/nvidia/GLCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nv/ComputeCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nvidia-omniverse/logs && \
    mkdir -p ${DOCKER_USER_HOME}/.local/share/ov/data && \
    mkdir -p ${DOCKER_USER_HOME}/Documents && \
    printf "pcm.!default {\\n  type null\\n}\\n\\nctl.!default {\\n  type null\\n}\\n" > ${DOCKER_USER_HOME}/.asoundrc

RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh --install

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y quadprog
RUN ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install --upgrade --no-cache-dir "numpy<2" opencv-python PyQt5==5.15.10

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo \"$UBUNTU_CODENAME\") main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS2_DISTRO}-${ROS2_APT_PACKAGE} \
    ros-${ROS2_DISTRO}-vision-msgs \
    ros-${ROS2_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS2_DISTRO}-rmw-fastrtps-cpp \
    ros-dev-tools && \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py rosdep ${ISAACLAB_PATH}/source && \
    echo "source /opt/ros/${ROS2_DISTRO}/setup.bash" >> ${HOME}/.bashrc

RUN mkdir -p ${DOCKER_USER_HOME}/.ros && \
    cp -r ${ISAACLAB_PATH}/docker/.ros/. ${DOCKER_USER_HOME}/.ros/

RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> ${HOME}/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard'" >> ${HOME}/.bashrc && \
    echo "export ENABLE_EGL=1" >> ${HOME}/.bashrc && \
    echo "export KIT_ENABLE_VULKAN_HEADLESS=1" >> ${HOME}/.bashrc && \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc && \
    echo "shopt -s histappend" >> /root/.bashrc && \
    echo "PROMPT_COMMAND='history -a'" >> /root/.bashrc

VOLUME [ \
    "/isaac-sim/kit/cache", \
    "/root/.cache/ov", \
    "/root/.cache/pip", \
    "/root/.cache/nvidia/GLCache", \
    "/root/.nv/ComputeCache", \
    "/root/.nvidia-omniverse/logs", \
    "/root/.local/share/ov/data", \
    "/workspace/isaaclab/docs/_build", \
    "/workspace/isaaclab/logs", \
    "/workspace/isaaclab/data_storage" \
]

WORKDIR /workspace
# -----------------------------------------------------
# TODO： 减小本地化size
RUN apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*
# TODO： 如果走了代理、但是想镜像本地化到其它机器，记得清空代理（或者容器内unset）
# ENV http_proxy=
# ENV https_proxy=
# ENV no_proxy=

ENTRYPOINT ["/bin/bash"]
