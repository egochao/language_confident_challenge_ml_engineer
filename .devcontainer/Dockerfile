FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime as base_image


ARG USERNAME=container
ARG USER_UID=500
ARG USER_GID=$USER_UID
ARG WORK_DIR=/home/$USERNAME

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --create-home --shell /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME
WORKDIR $WORK_DIR

COPY --chown=$USERNAME . .

ENV VENV_PATH=$WORK_DIR/venv
ENV PATH="$VENV_PATH/bin:$PATH"
RUN python -m venv $VENV_PATH
RUN pip install -r requirements.txt
