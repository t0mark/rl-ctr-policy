# syntax=docker/dockerfile:1

# Official prebuilt Isaac Sim + Isaac Lab image from NVIDIA NGC
ARG ISAACLAB_VERSION=2.3.2
FROM nvcr.io/nvidia/isaac-lab:${ISAACLAB_VERSION}

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y

# Project workspace is bind-mounted here at runtime (see docker-compose.yml)
WORKDIR /workspace/dreamwaq

ENTRYPOINT ["bash"]
