FROM cjber/dots
ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /root

# install cuda
RUN pacman -Syu --noconfirm && pacman -S cuda nvidia --noconfirm
USER cjber
WORKDIR /home/cjber
RUN git clone https://aur.archlinux.org/libnvidia-container.git && \
        cd libnvidia-container && \
        makepkg -si --noconfirm && \
        sudo pacman -U libnvidia-container*.pkg.* --noconfirm && \
        cd && \
        git clone https://aur.archlinux.org/nvidia-container-toolkit.git && \
        cd nvidia-container-toolkit && \
        makepkg -si --noconfirm && \
        sudo pacman -U nvidia-container-toolkit*.pkg.* --noconfirm

USER root
WORKDIR /root
RUN git clone https://github.com/cjber/ger_wiki.git
WORKDIR ger_wiki
RUN poetry install

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441"
