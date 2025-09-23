cmake_minimum_required(VERSION 3.16)
project(HeatEquationPINN.Cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug und Release Konfigurationen definieren
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

# CPU-optimized build settings
# Mach es so:
# mkdir build
# cd build
# cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="C:/Users/haasr/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_PREFIX_PATH="C:/libs/libtorch" ..
if(MSVC)
    # Visual Studio optimizations for CPU
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /Oy /GL /fp:fast")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    
    # Enable OpenMP for Visual Studio
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    add_definitions(-DUSE_OPENMP)
    
    # Additional CPU optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -ffast-math")
    
    # Find OpenMP for other compilers
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_definitions(-DUSE_OPENMP)
    endif()
endif()
# Setze vcpkg toolchain explizit
set(CMAKE_TOOLCHAIN_FILE "C:/Users/haasr/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# Finde Torch
find_package(Torch REQUIRED)

# Erstelle executable
add_executable(HeatEquationPINN.Cpp main.cpp)

# Include-Verzeichnis hinzufuegen
target_include_directories(HeatEquationPINN.Cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(HeatEquationPINN.Cpp ${TORCH_LIBRARIES})

# Compiler flags
target_compile_features(HeatEquationPINN.Cpp PRIVATE cxx_std_23)

# Debug-Konfiguration
target_compile_options(HeatEquationPINN.Cpp PRIVATE
    $<$<CONFIG:Debug>:
        $<$<CXX_COMPILER_ID:MSVC>:/Od /Zi /MDd>
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O0 -g>
    >
)

# Release-Konfiguration
target_compile_options(HeatEquationPINN.Cpp PRIVATE
    $<$<CONFIG:Release>:
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG /MD>
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -DNDEBUG>
    >
)

# Windows spezifisch
if(MSVC)
    target_compile_options(HeatEquationPINN.Cpp PRIVATE /bigobj)
endif()