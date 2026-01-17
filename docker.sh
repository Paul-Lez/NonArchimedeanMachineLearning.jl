#!/bin/bash
# Convenience script for Docker operations

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "NAML Docker Helper Script"
    echo ""
    echo "Usage: ./docker.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Docker image"
    echo "  rebuild       Rebuild the Docker image (no cache)"
    echo "  repl          Start interactive Julia REPL"
    echo "  jupyter       Start Jupyter notebook server"
    echo "  test [file]   Run a test file (e.g., test/polydisc.jl)"
    echo "  run [file]    Run a Julia script"
    echo "  shell         Start bash shell in container"
    echo "  clean         Stop and remove containers"
    echo "  deep-clean    Remove containers and images"
    echo ""
    echo "Examples:"
    echo "  ./docker.sh build"
    echo "  ./docker.sh repl"
    echo "  ./docker.sh test test/polydisc.jl"
    echo "  ./docker.sh run my_script.jl"
}

case "$1" in
    build)
        echo -e "${BLUE}Building Docker image...${NC}"
        docker-compose build
        echo -e "${GREEN}Build complete!${NC}"
        ;;

    rebuild)
        echo -e "${BLUE}Rebuilding Docker image (no cache)...${NC}"
        docker-compose build --no-cache
        echo -e "${GREEN}Rebuild complete!${NC}"
        ;;

    repl)
        echo -e "${BLUE}Starting Julia REPL...${NC}"
        echo -e "${YELLOW}Tip: Use 'using NAML' to load the package${NC}"
        docker-compose run --rm naml
        ;;

    jupyter)
        echo -e "${BLUE}Starting Jupyter notebook server...${NC}"
        echo -e "${YELLOW}Access at http://localhost:8888${NC}"
        docker-compose up jupyter
        ;;

    test)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Usage: ./docker.sh test [file]${NC}"
            echo "Example: ./docker.sh test test/polydisc.jl"
            exit 1
        fi
        echo -e "${BLUE}Running test: $2${NC}"
        docker-compose run --rm naml julia --project=. "$2"
        ;;

    run)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Usage: ./docker.sh run [file]${NC}"
            echo "Example: ./docker.sh run my_script.jl"
            exit 1
        fi
        echo -e "${BLUE}Running script: $2${NC}"
        docker-compose run --rm naml julia --project=. "$2"
        ;;

    shell)
        echo -e "${BLUE}Starting bash shell in container...${NC}"
        docker-compose run --rm naml bash
        ;;

    clean)
        echo -e "${BLUE}Stopping and removing containers...${NC}"
        docker-compose down
        echo -e "${GREEN}Cleanup complete!${NC}"
        ;;

    deep-clean)
        echo -e "${YELLOW}This will remove all containers and images. Continue? (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Removing containers and images...${NC}"
            docker-compose down --rmi all --volumes
            echo -e "${GREEN}Deep cleanup complete!${NC}"
        else
            echo "Cancelled."
        fi
        ;;

    *)
        print_usage
        exit 1
        ;;
esac
