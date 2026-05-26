#!/usr/bin/env bash
# Reproducible setup for BevSplat live inference in this project.
# Run from the project root after activating your Python env.
#
# Prerequisites you have to install yourself (not scripted because they
# all need user consent or licensed downloads):
#   * CUDA toolkit 12.8
#   * MSVC C++ Build Tools (Windows: VS 2022 Build Tools w/ VCTools workload)
#   * torch 2.7.0+cu128, torchvision 0.22.0+cu128, xformers 0.0.30
#       pip install --force-reinstall --no-deps \
#         --index-url https://download.pytorch.org/whl/cu128 \
#         torch==2.7.0 torchvision==0.22.0
#       pip install --no-deps xformers==0.0.30
#   * lpips and addict
#       pip install lpips addict
#
# Then run this script.

set -euo pipefail

cd "$(dirname "$0")/.."   # project root

mkdir -p third_party

if [[ ! -d third_party/BevSplat ]]; then
    git clone --depth 1 https://github.com/wangqww/BevSplat third_party/BevSplat
fi

if [[ ! -d third_party/BevSplat/pano_feature_gaussian/third_party/glm ]]; then
    git clone --depth 1 https://github.com/g-truc/glm.git \
        third_party/BevSplat/pano_feature_gaussian/third_party/glm
fi
if [[ ! -d third_party/BevSplat/feature_gaussian/third_party/glm ]]; then
    git clone --depth 1 https://github.com/g-truc/glm.git \
        third_party/BevSplat/feature_gaussian/third_party/glm
fi

echo "Applying local patches (idempotent)..."
( cd third_party/BevSplat && \
  git apply --check ../../patches/bevsplat_local.patch 2>/dev/null && \
  git apply       ../../patches/bevsplat_local.patch ) \
  || echo "  (patches already applied or partially applied — skipping)"

mkdir -p third_party/BevSplat-weights
cat <<'EOF'

--- Manual download required ---

The KITTI/VIGOR checkpoints are not in this script. Download
KITTI_no_GPS.pth from the authors' OneDrive share into
third_party/BevSplat-weights/ — see the BevSplat section in README.md
for the link and the full checkpoint table.

--- Then build the CUDA extensions ---

On Windows, run third_party/build_extensions.bat from a regular
command prompt (it self-activates vcvars64). On Linux/macOS you can
build manually:

    cd third_party/BevSplat/feature_gaussian      && pip install -e .
    cd third_party/BevSplat/pano_feature_gaussian && pip install -e .

Then test:

    python main.py --skip-download \
        --enable-bev-splat \
        --bev-splat-weights third_party/BevSplat-weights/KITTI_no_GPS.pth \
        --bev-splat-repo-path third_party/BevSplat
EOF
