cmake_minimum_required(VERSION 3.15)
project(flexipipe_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimize for speed
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_BUILD_TYPE Release)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    flexipipe_cpp.cpp
    vocab_loader.cpp
    tokenizer.cpp
    normalizer.cpp
    contractions.cpp
    viterbi_optimized.cpp
    io_conllu.cpp
    io_teitok.cpp
    io_teitok_writer.cpp
)

# Use FetchContent for dependencies (modern CMake approach)
include(FetchContent)

# Add rapidjson (header-only library)
FetchContent_Declare(
    rapidjson
    URL https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz
)
FetchContent_MakeAvailable(rapidjson)
include_directories(${rapidjson_SOURCE_DIR}/include)

# Add pugixml (needs to be compiled)
find_package(pugixml QUIET)
if(NOT pugixml_FOUND)
    FetchContent_Declare(
        pugixml
        URL https://github.com/zeux/pugixml/archive/refs/tags/v1.14.tar.gz
    )
    set(PUGIXML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PUGIXML_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(pugixml)
endif()

# Executable
add_executable(flexipipe_cpp ${SOURCES})

# Link pugixml
if(pugixml_FOUND)
    target_link_libraries(flexipipe_cpp PRIVATE pugixml::pugixml)
else()
    target_link_libraries(flexipipe_cpp PRIVATE pugixml)
endif()

# Link libraries (if needed)
# target_link_libraries(flexipipe_cpp ...)

# Install
install(TARGETS flexipipe_cpp DESTINATION bin)


