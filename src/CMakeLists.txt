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

# Add rapidjson (header-only)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/rapidjson/include)

# Add pugixml (needs to be compiled)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pugixml/src)
set(PUGIXML_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pugixml/src/pugixml.cpp)

# Executable
add_executable(flexipipe_cpp ${SOURCES} ${PUGIXML_SOURCE})

# Link libraries (if needed)
# target_link_libraries(flexipipe_cpp ...)

# Install
install(TARGETS flexipipe_cpp DESTINATION bin)


