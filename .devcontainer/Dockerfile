FROM "tensorflow/tensorflow:latest-gpu"
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN \
    apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    && apt-get -y install git procps lsb-release ffmpeg libsm6 libxext6 \
    && pip install --upgrade pip \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && echo 'PROMPT_DIRTRIM=1' >> "/home/$USERNAME/.bashrc"

ENV DEBIAN_FRONTEND=