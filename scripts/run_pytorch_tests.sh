#!/bin/bash
set -e

echo "==================================================="
echo " WarpKV PyTorch Integration Build & Test Script"
echo "==================================================="

# 1. Check for nvcc and python
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc (CUDA Toolkit) is not installed or not in PATH."
    exit 1
fi

if ! python3 -c "import torch" &> /dev/null; then
    echo "❌ ERROR: PyTorch is not installed. Please run: pip install torch"
    exit 1
fi

# 2. Get PyTorch CMake Prefix Path
TORCH_CMAKE_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
echo "✅ Found PyTorch CMake config at: $TORCH_CMAKE_PATH"

# 3. Build the extension
echo "🔨 Building warpkv_torch extension..."
mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH" -DCMAKE_BUILD_TYPE=Release ..
make warpkv_torch -j$(nproc)

echo "✅ Build complete."

# 4. Copy the compiled .so object up to the tests/integration directory
# so Python can easily find 'warpkv_torch' when importing
find . -name "warpkv_torch*.so" -exec cp {} ../tests/integration/ \;
cd ..

# 5. Run the integration tests
echo "🧪 Running PyTorch Integration Tests..."
cd tests/integration
python3 test_torch_integration.py

echo "🎉 All PyTorch tests passed successfully!"
