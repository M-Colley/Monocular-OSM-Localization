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

# --- Windows/macOS case-collision guard for the DINO backbone ----------
# Upstream tracks two files differing only in case: models/dino_fit.py and
# models/dino_Fit.py. On a case-insensitive filesystem (Windows NTFS, macOS
# APFS) they map to ONE physical file, and git checks out dino_fit.py last
# (byte order 'F' < 'f'), so the file our patch targets (dino_fit.py) is the
# one that lands on disk — the patch applies cleanly on a fresh clone.
#
# The patch deliberately does NOT carry a dino_Fit.py hunk. `git apply`
# can't patch the colliding twin: its expected "before" content (the old
# upstream dino_Fit.py) never matches the shared physical file, so a
# two-file patch fails `git apply --check` outright on Windows. (Verified.)
#
# We instead (a) unify git's index so both names point at the patched blob,
# which keeps `git status` clean and blunts accidental `git checkout`/`stash`
# reverts, and (b) hard-verify the physical file actually carries the
# patched github loader before declaring success — this catches the exact
# failure mode where a stray git op reverts the shared file to the
# hardcoded Linux `/home/qiwei/...` path.
( cd third_party/BevSplat && \
  git add models/dino_fit.py models/dino_Fit.py >/dev/null 2>&1 || true )

if ! grep -q 'ywyue/FiT3D' third_party/BevSplat/models/dino_fit.py; then
    echo "ERROR: third_party/BevSplat/models/dino_fit.py is missing the" >&2
    echo "       patched 'ywyue/FiT3D' loader — the Windows/macOS case" >&2
    echo "       collision likely reverted it. Re-run:" >&2
    echo "         (cd third_party/BevSplat && git apply ../../patches/bevsplat_local.patch)" >&2
    echo "       and do NOT run git stash/reset/checkout inside that clone." >&2
    exit 1
fi
echo "  DINO loader OK (ywyue/FiT3D github, case-collision twin unified)"

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
