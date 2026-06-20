#!/bin/bash

# FHEON Docker control and execution helper script

IMAGE_NAME="fheon"
CONTAINER_NAME="fheon_container"

show_help() {
    echo "FHEON Docker Execution Helper"
    echo "============================="
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  build            - Build the Docker image (compiles OpenFHE and FHEON)"
    echo "  build-nocache    - Build the Docker image without cache (forces fresh clone)"
    echo "  run              - Start an interactive bash session in the container"
    echo "  run-lenet5       - Run LeNet5 model in Docker"
    echo "  run-resnet20     - Run ResNet20Optimized model in Docker"
    echo "  run-resnet34     - Run ResNet34Optimized model in Docker"
    echo "  run-vgg11        - Run VGG11 model in Docker"
    echo "  run-vgg16        - Run VGG16 model in Docker"
    echo "  run-accuracy     - Run the python accuracy verification script in Docker"
    echo "  clean            - Remove any stopped FHEON Docker containers"
    echo "  clean-image      - Remove FHEON containers, image, and build cache completely"
    echo "  help             - Show this help message"
    echo ""
    echo "Options:"
    echo "  --test_size <n>  - Specify the test dataset size to run (applicable to run-* model commands)"
    echo ""
}

# Parse command-line options
TEST_SIZE=""
for ((i=1; i<=$#; i++)); do
    if [ "${!i}" = "--test_size" ]; then
        next_idx=$((i+1))
        TEST_SIZE="${!next_idx}"
    elif [[ "${!i}" == --test_size=* ]]; then
        TEST_SIZE="${!i#*=}"
    fi
done

case "$1" in
    build)
        echo "Pulling latest changes from git..."
        git pull
        echo "Building Docker image: $IMAGE_NAME..."
        docker build -t "$IMAGE_NAME" .
        ;;
    build-nocache)
        echo "Pulling latest changes from git..."
        git pull
        echo "Building Docker image without cache: $IMAGE_NAME..."
        docker build --no-cache -t "$IMAGE_NAME" .
        ;;

    run)
        echo "Starting interactive bash session in container..."
        docker run --rm -it --init --name "$CONTAINER_NAME" "$IMAGE_NAME"
        ;;
    run-lenet5)
        docker run --rm -it --init "$IMAGE_NAME" ./LeNet5 ${TEST_SIZE:+--test_size "$TEST_SIZE"}
        ;;
    run-resnet20)
        docker run --rm -it --init "$IMAGE_NAME" ./ResNet20Optimized ${TEST_SIZE:+--test_size "$TEST_SIZE"}
        ;;
    run-resnet34)
        docker run --rm -it --init "$IMAGE_NAME" ./ResNet34Optimized ${TEST_SIZE:+--test_size "$TEST_SIZE"}
        ;;
    run-vgg11)
        docker run --rm -it --init "$IMAGE_NAME" ./VGG11 ${TEST_SIZE:+--test_size "$TEST_SIZE"}
        ;;
    run-vgg16)
        docker run --rm -it --init "$IMAGE_NAME" ./VGG16 ${TEST_SIZE:+--test_size "$TEST_SIZE"}
        ;;
    run-accuracy)
        docker run --rm -it --init "$IMAGE_NAME" bash -c "cd /app/results && python3 accuracy.py"
        ;;
    clean)
        echo "Removing any stopped FHEON containers..."
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        ;;
    clean-image)
        echo "Removing FHEON container: $CONTAINER_NAME..."
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        echo "Pruning stopped containers..."
        docker container prune -f
        echo "Removing Docker image: $IMAGE_NAME..."
        docker rmi -f "$IMAGE_NAME" 2>/dev/null || true
        echo "Pruning Docker build cache (forces fresh clone on next build)..."
        docker builder prune -f
        ;;
    help|*)
        show_help
        ;;
esac

