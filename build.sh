#!/bin/sh

USAGE="Usage: $0 [--debug|--release]"

BUILD_DIR="build"

BUILD_TYPE=Release
SANITIZE=FALSE

if [ $# -eq 0 ]; then
    COLOR='\033[36m'
    BOLD='\033[1m'
    BOLD_OFF='\033[22m'
    COLOR_OFF='\033[0m'

    echo -e "${COLOR}Warning: by default the build will run in ${BOLD}RELEASE${BOLD_OFF} mode.${COLOR_OFF}"
fi
    
while [ $# -gt 0 ]; do
    case "$1" in
        --release)
            BUILD_TYPE=Release
            SANITIZE=FALSE
            ;;

        --debug)
            BUILD_TYPE=Debug
            SANITIZE=TRUE
            ;;

        *)
            echo "Unknown option: $1"
            echo "$USAGE"
            exit 1
            ;;
    esac
    shift
done

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DSANITIZE="$SANITIZE" ..
cmake --build .

