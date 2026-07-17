#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $SCRIPT_DIR

# For Qt and it's backend
sudo -E apt -y update
sudo -E apt -y install \
    libxcb-xinerama0 \
    libqt5gui5 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-util1 \
    libxcb-xkb1 \
    libxkbcommon-x11-0

sudo -E apt -y install \
    python3-virtualenv \
    imagemagick \
    python3-pip \
    ipython3

PREFIX=$HOME/.local/opt/labelme/
mkdir -p $PREFIX
virtualenv $PREFIX

source $PREFIX/bin/activate

pip install --upgrade pip

# install in editable mode
pip install -e .

mkdir -p ~/.local/bin/
cat > ~/.local/bin/labelme_yolo << EOF
#!/bin/bash
source ~/.local/opt/labelme/bin/activate
labelme \$@
exit \$?
EOF
chmod a+x ~/.local/bin/labelme_yolo



