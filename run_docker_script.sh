#!/bin/bash

# FHEON Docker control and execution helper script

IMAGE_NAME="fheon"
CONTAINER_NAME="fheon_container"

show_help() {
    echo "FHEON Docker Execution Helper"
    echo "============================="
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build            - Build the Docker image (compiles OpenFHE and FHEON)"
    echo "  run              - Start an interactive bash session in the container"
    echo "  run-lenet5       - Run LeNet5 model in Docker"
    echo "  run-resnet20     - Run ResNet20Optimized model in Docker"
    echo "  run-resnet34     - Run ResNet34Optimized model in Docker"
    echo "  run-vgg11        - Run VGG11 model in Docker"
    echo "  run-vgg16        - Run VGG16 model in Docker"
    echo "  run-accuracy     - Run the python accuracy verification script in Docker"
    echo "  clean            - Remove any stopped FHEON Docker containers"
    echo "  help             - Show this help message"
    echo ""
}

case "$1" in
    build)
        echo "Building Docker image: $IMAGE_NAME..."
        docker build -t "$IMAGE_NAME" .
        ;;
    run)
        echo "Starting interactive bash session in container..."
        docker run --rm -it --init --name "$CONTAINER_NAME" "$IMAGE_NAME"
        ;;
    run-lenet5)
        docker run --rm -it --init "$IMAGE_NAME" ./LeNet5
        ;;
    run-resnet20)
        docker run --rm -it --init "$IMAGE_NAME" ./ResNet20Optimized
        ;;
    run-resnet34)
        docker run --rm -it --init "$IMAGE_NAME" ./ResNet34Optimized
        ;;
    run-vgg11)
        docker run --rm -it --init "$IMAGE_NAME" ./VGG11
        ;;
    run-vgg16)
        docker run --rm -it --init "$IMAGE_NAME" ./VGG16
        ;;
    run-accuracy)
        docker run --rm -it --init "$IMAGE_NAME" bash -c "cd /app/results && python3 accuracy.py"
        ;;
    clean)
        echo "Removing any stopped FHEON containers..."
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        ;;
    help|*)
        show_help
        ;;
esac
