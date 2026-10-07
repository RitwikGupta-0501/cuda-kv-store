import re

with open('CMakeLists.txt', 'r') as f:
    c = f.read()

# Add Torch integration
torch_cmake = """
# PyTorch Extension (Phase 4)
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "Found PyTorch: ${TORCH_VERSION}, building warpkv_torch")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    
    add_library(warpkv_torch MODULE src/python/warpkv_torch.cpp)
    target_link_libraries(warpkv_torch PRIVATE warpkv_engine ${TORCH_LIBRARIES})
    
    # Configure extension name macro for Pybind11
    target_compile_definitions(warpkv_torch PRIVATE TORCH_EXTENSION_NAME=warpkv_torch)
    set_target_properties(warpkv_torch PROPERTIES PREFIX "" SUFFIX ".so")
else()
    message(WARNING "PyTorch not found, skipping warpkv_torch build.")
endif()
"""

if 'warpkv_torch' not in c:
    c = c + '\n' + torch_cmake

with open('CMakeLists.txt', 'w') as f:
    f.write(c)
