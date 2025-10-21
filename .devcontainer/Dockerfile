FROM gcr.io/kaggle-gpu-images/python:v161

COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# # --------- pytorch --------- #
# RUN uv pip install --system torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
RUN uv pip install --system lightning>=2.0.0
RUN uv pip install --system torchmetrics>=0.11.4

# # --------- hydra --------- #
RUN uv pip install --system hydra-core==1.3.2
RUN uv pip install --system hydra-colorlog==1.2.0
RUN uv pip install --system hydra-optuna-sweeper==1.2.0

# # --------- others --------- #
RUN uv pip install --system rootutils       # standardizing the project root setup
RUN uv pip install --system pre-commit      # hooks for applying linters on commit
RUN uv pip install --system rich            # beautiful text formatting in terminal
RUN uv pip install --system pytest          # tests
RUN uv pip install --system build

# # --------- tensorrt --------- #
RUN uv pip install --system tensorrt-cu12==10.7.0.post1
RUN uv pip install --system torch-tensorrt==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
RUN uv pip install --system triton==3.2.0

# # --------- rsna --------- #
RUN uv pip install --system nibabel pydicom
RUN uv pip install --system napari[all]
RUN uv pip install --system pylibjpeg pylibjpeg-libjpeg python-gdcm 
RUN uv pip install --system h5py zarr
RUN uv pip install --system monai 
# surfa requires numpy<2.0
# RUN uv pip install --system surfa

# for napari
RUN apt-get install -y libegl1 libdbus-1-3 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 \
    x11-utils libxcb-cursor0 libopengl0 libegl1-mesa

RUN uv pip install --force-reinstall numpy==1.26.4
RUN apt install -y dcm2niix libgdcm-tools 
RUN uv pip install --system git+https://github.com/MIC-DKFZ/batchgeneratorsv2.git@07541d7eb5a4839aa4a5e494a123f3fe69ccfd4f

# # --------- Codex ---------- #
RUN npm install -g @openai/codex

ENV PIP_LOG=/tmp/pip.log

ARG USERNAME=tomo
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && usermod -aG sudo $USERNAME

RUN touch /tmp/kaggle.log \
    && chmod 666 /tmp/kaggle.log \
    && chown $USERNAME:$USERNAME /tmp/kaggle.log

RUN mkdir -p /home/$USERNAME/.kaggle
COPY kaggle.json /home/$USERNAME/.kaggle/kaggle.json
RUN chmod 600 /home/$USERNAME/.kaggle/kaggle.json \
    && chown $USERNAME:$USERNAME /home/$USERNAME/.kaggle/kaggle.json

# Configure pip to install to user site for non-root user
RUN mkdir -p /home/$USERNAME/.config/pip \
    && printf "[install]\nuser = true\n" > /home/$USERNAME/.config/pip/pip.conf \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.config

RUN sudo mkdir /kaggle \
    && sudo chown $USERNAME:$USERNAME /kaggle

ENV PYTHONUSERBASE=/home/${USERNAME}/.local
ENV PATH=${PYTHONUSERBASE}/bin:$PATH

ENV nnUNet_raw=/workspace/data/nnUNet/nnUNet_raw
ENV nnUNet_preprocessed=/workspace/data/nnUNet/nnUNet_preprocessed
ENV nnUNet_results=/workspace/logs/nnUNet_results

WORKDIR /workspace