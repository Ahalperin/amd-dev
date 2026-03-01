#!/usr/bin/env bash
#
# Build RCCL Docker image from the rocm-systems source-of-truth repository
# with sparse checkout for projects/rccl and projects/rccl-tests.

set -euo pipefail

# Default values
ROCM_VERSION="7.2"
ROCM_IMAGE_NAME="rocm/dev-ubuntu-22.04"
ROCM_IMAGE_TAG=""  # Will be set to ${ROCM_VERSION}-complete if not provided
ROCM_SYSTEMS_REPO="https://github.com/ROCm/rocm-systems.git"
ROCM_SYSTEMS_BRANCH="develop"
GPU_TARGETS="gfx950"
DOCKERFILE="Dockerfile.rccl_from_rocm_systems"
IMAGE_NAME="rccl-build"
IMAGE_TAG=""  # Will be constructed from ROCM_VERSION and ROCM_SYSTEMS_BRANCH
NO_CACHE=false

print_usage() {
    cat << 'EOF'
Usage:
  ./build_rccl_from_rocm_systems.sh [OPTIONS]

Options:
  --rocm-version VERSION         ROCm version (default: 7.2)
  --rocm-image-name NAME         ROCm base image name (default: rocm/dev-ubuntu-22.04)
  --rocm-image-tag TAG           ROCm base image tag (default: ${ROCM_VERSION}-complete)
  --rocm-systems-repo URL        rocm-systems repository URL (default: https://github.com/ROCm/rocm-systems.git)
  --rocm-systems-branch BRANCH   rocm-systems branch to build (default: develop)
  --gpu-targets TARGETS          GPU targets (default: gfx950)
  --dockerfile PATH              Dockerfile path (default: Dockerfile.rccl_from_rocm_systems)
  --image-name NAME              Docker image name (default: rccl-build)
  --image-tag TAG                Docker image tag (default: from ROCM_VERSION + rocm-systems branch)
  --no-cache                     Build without using cache
  --help, -h                     Show this help message

Examples:
  # Build with default settings
  ./build_rccl_from_rocm_systems.sh

  # Build from a custom rocm-systems branch
  ./build_rccl_from_rocm_systems.sh --rocm-systems-branch release/rocm-rel-7.1

  # Build with specific ROCm image tag and multiple GPU targets
  ./build_rccl_from_rocm_systems.sh \
    --rocm-image-tag 7.1.1-complete \
    --gpu-targets "gfx942;gfx950"
EOF
}

sanitize_for_tag() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g'
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocm-version)
            ROCM_VERSION="$2"
            shift 2
            ;;
        --rocm-image-name)
            ROCM_IMAGE_NAME="$2"
            shift 2
            ;;
        --rocm-image-tag)
            ROCM_IMAGE_TAG="$2"
            shift 2
            ;;
        --rocm-systems-repo)
            ROCM_SYSTEMS_REPO="$2"
            shift 2
            ;;
        --rocm-systems-branch)
            ROCM_SYSTEMS_BRANCH="$2"
            shift 2
            ;;
        --gpu-targets)
            GPU_TARGETS="$2"
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option '$1'"
            echo ""
            print_usage
            exit 1
            ;;
    esac
done

# Set default ROCM_IMAGE_TAG if not provided
if [[ -z "$ROCM_IMAGE_TAG" ]]; then
    ROCM_IMAGE_TAG="${ROCM_VERSION}-complete"
fi

# Set default IMAGE_TAG if not provided
if [[ -z "$IMAGE_TAG" ]]; then
    ROCM_VERSION_SANITIZED="$(sanitize_for_tag "$ROCM_VERSION")"
    ROCM_SYSTEMS_BRANCH_SANITIZED="$(sanitize_for_tag "$ROCM_SYSTEMS_BRANCH")"
    IMAGE_TAG="${ROCM_VERSION_SANITIZED}-${ROCM_SYSTEMS_BRANCH_SANITIZED}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$SCRIPT_DIR/$DOCKERFILE" ]]; then
    echo "ERROR: Dockerfile '$SCRIPT_DIR/$DOCKERFILE' not found"
    exit 1
fi

echo "=========================================="
echo "Building RCCL Docker Image (rocm-systems)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  ROCm Version:          $ROCM_VERSION"
echo "  ROCm Image Name:       $ROCM_IMAGE_NAME"
echo "  ROCm Image Tag:        $ROCM_IMAGE_TAG"
echo "  rocm-systems Repo:     $ROCM_SYSTEMS_REPO"
echo "  rocm-systems Branch:   $ROCM_SYSTEMS_BRANCH"
echo "  Sparse Paths:          projects/rccl, projects/rccl-tests"
echo "  GPU Targets:           $GPU_TARGETS"
echo "  Dockerfile:            $DOCKERFILE"
echo "  Image Name:            $IMAGE_NAME"
echo "  Image Tag:             $IMAGE_TAG"
echo "  No Cache:              $NO_CACHE"
echo ""
echo "=========================================="
echo ""

echo "[1/1] Building Docker image..."
docker_build_cmd=(
    docker build
    --build-arg "ROCM_VERSION=$ROCM_VERSION"
    --build-arg "ROCM_IMAGE_NAME=$ROCM_IMAGE_NAME"
    --build-arg "ROCM_IMAGE_TAG=$ROCM_IMAGE_TAG"
    --build-arg "ROCM_SYSTEMS_REPO=$ROCM_SYSTEMS_REPO"
    --build-arg "ROCM_SYSTEMS_BRANCH=$ROCM_SYSTEMS_BRANCH"
    --build-arg "GPU_TARGETS=$GPU_TARGETS"
    -f "$SCRIPT_DIR/$DOCKERFILE"
    -t "${IMAGE_NAME}:${IMAGE_TAG}"
    "$SCRIPT_DIR"
)

if [[ "$NO_CACHE" == "true" ]]; then
    docker_build_cmd=(docker build --no-cache "${docker_build_cmd[@]:2}")
fi

"${docker_build_cmd[@]}"

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run the container:"
echo "  docker run -it --device=/dev/kfd --device=/dev/dri \\"
echo "    --security-opt seccomp=unconfined \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
