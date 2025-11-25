cmake_minimum_required(VERSION 3.15)
project(flexitag LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

include(FetchContent)

add_library(flexitag
    src/settings.cpp
    src/lexicon.cpp
    src/io_teitok.cpp
    src/tagger.cpp
    src/normalizer.cpp
)

target_include_directories(flexitag
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Try system pugixml first
find_package(pugixml QUIET)

if(pugixml_FOUND)
    target_link_libraries(flexitag PRIVATE pugixml::pugixml)
else()
    find_path(PUGIXML_INCLUDE_DIR pugixml.hpp)
    find_library(PUGIXML_LIBRARY NAMES pugixml)
    if(PUGIXML_INCLUDE_DIR AND PUGIXML_LIBRARY)
        target_include_directories(flexitag PRIVATE ${PUGIXML_INCLUDE_DIR})
        target_link_libraries(flexitag PRIVATE ${PUGIXML_LIBRARY})
    else()
        message(STATUS "System pugixml not found, fetching local copy")
        FetchContent_Declare(
            pugixml_fetch
            URL https://github.com/zeux/pugixml/archive/refs/tags/v1.14.tar.gz
        )
        set(PUGIXML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(PUGIXML_BUILD_SHARED OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(pugixml_fetch)
        target_link_libraries(flexitag PRIVATE pugixml)
    endif()
endif()

find_package(nlohmann_json CONFIG QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(flexitag PRIVATE nlohmann_json::nlohmann_json)
else()
    FetchContent_Declare(
        nlohmann_json_fetch
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json_fetch)
    target_link_libraries(flexitag PRIVATE nlohmann_json::nlohmann_json)
endif()

# ICU (International Components for Unicode) for proper Unicode handling
# Supports UTF-8, UTF-16, UTF-32 and all Unicode operations
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ICU icu-uc icu-i18n)
endif()

if(ICU_FOUND)
    target_include_directories(flexitag PRIVATE ${ICU_INCLUDE_DIRS})
    target_link_libraries(flexitag PRIVATE ${ICU_LIBRARIES})
    target_compile_definitions(flexitag PRIVATE FLEXITAG_USE_ICU=1)
else()
    # Try CMake's FindICU
    find_package(ICU QUIET)
    if(ICU_FOUND)
        target_link_libraries(flexitag PRIVATE ${ICU_LIBRARIES})
        target_include_directories(flexitag PRIVATE ${ICU_INCLUDE_DIRS})
        target_compile_definitions(flexitag PRIVATE FLEXITAG_USE_ICU=1)
    else()
        # Try to find ICU manually (common on macOS with Homebrew)
        # Homebrew installs ICU in versioned directories like /opt/homebrew/Cellar/icu4c@77/77.1
        file(GLOB ICU_CELLAR_DIRS "/opt/homebrew/Cellar/icu4c*")
        list(APPEND ICU_SEARCH_PATHS
            /opt/homebrew/include
            /usr/local/include
            /usr/include
        )
        foreach(cellar_dir ${ICU_CELLAR_DIRS})
            file(GLOB version_dirs "${cellar_dir}/*")
            foreach(version_dir ${version_dirs})
                if(IS_DIRECTORY "${version_dir}/include")
                    list(APPEND ICU_SEARCH_PATHS "${version_dir}/include")
                endif()
            endforeach()
        endforeach()
        
        find_path(ICU_INCLUDE_DIR unicode/unistr.h
            PATHS ${ICU_SEARCH_PATHS}
        )
        
        set(ICU_LIB_SEARCH_PATHS
            /opt/homebrew/lib
            /usr/local/lib
            /usr/lib
        )
        foreach(cellar_dir ${ICU_CELLAR_DIRS})
            file(GLOB version_dirs "${cellar_dir}/*")
            foreach(version_dir ${version_dirs})
                if(IS_DIRECTORY "${version_dir}/lib")
                    list(APPEND ICU_LIB_SEARCH_PATHS "${version_dir}/lib")
                endif()
            endforeach()
        endforeach()
        
        find_library(ICU_UC_LIBRARY
            NAMES icuuc libicuuc icuuc.76 icuuc.77 icuuc.78
            PATHS ${ICU_LIB_SEARCH_PATHS}
        )
        find_library(ICU_I18N_LIBRARY
            NAMES icui18n libicui18n icui18n.76 icui18n.77 icui18n.78
            PATHS ${ICU_LIB_SEARCH_PATHS}
        )
        if(ICU_INCLUDE_DIR AND ICU_UC_LIBRARY AND ICU_I18N_LIBRARY)
            # Verify the include directory actually has the header file
            if(NOT EXISTS "${ICU_INCLUDE_DIR}/unicode/unistr.h")
                # Try to find the correct path
                foreach(search_path ${ICU_SEARCH_PATHS})
                    if(EXISTS "${search_path}/unicode/unistr.h")
                        set(ICU_INCLUDE_DIR "${search_path}")
                        break()
                    endif()
                endforeach()
            endif()
            if(EXISTS "${ICU_INCLUDE_DIR}/unicode/unistr.h")
                target_include_directories(flexitag PRIVATE ${ICU_INCLUDE_DIR})
                target_link_libraries(flexitag PRIVATE ${ICU_UC_LIBRARY} ${ICU_I18N_LIBRARY})
                target_compile_definitions(flexitag PRIVATE FLEXITAG_USE_ICU=1)
                message(STATUS "Found ICU: ${ICU_INCLUDE_DIR}, ${ICU_UC_LIBRARY}, ${ICU_I18N_LIBRARY}")
            else()
                message(FATAL_ERROR "ICU headers not found in ${ICU_INCLUDE_DIR}/unicode/unistr.h")
            endif()
        else()
            message(FATAL_ERROR "ICU (International Components for Unicode) is required but not found. Please install ICU development packages:\n  macOS: brew install icu4c\n  Ubuntu/Debian: sudo apt-get install libicu-dev\n  Fedora: sudo dnf install libicu-devel")
        endif()
    endif()
endif()

add_executable(flexitag_cli cli/flexitag_cli.cpp)
target_link_libraries(flexitag_cli PRIVATE flexitag)

option(FLEXITAG_BUILD_PYTHON "Build Python bindings" ON)
if(FLEXITAG_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(flexitag_py bindings/python/module.cpp)
        target_include_directories(flexitag_py PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(flexitag_py PRIVATE flexitag)
        # Python module needs ICU headers (inherited from flexitag, but ensure they're available)
        if(ICU_INCLUDE_DIR)
            target_include_directories(flexitag_py PRIVATE ${ICU_INCLUDE_DIR})
        endif()
    else()
        message(STATUS "pybind11 not found, fetching local copy")
        include(FetchContent)
        FetchContent_Declare(
            pybind11_fetch
            URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
        )
        FetchContent_MakeAvailable(pybind11_fetch)
        pybind11_add_module(flexitag_py bindings/python/module.cpp)
        target_include_directories(flexitag_py PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(flexitag_py PRIVATE flexitag)
        # Python module needs ICU headers (inherited from flexitag, but ensure they're available)
        if(ICU_INCLUDE_DIR)
            target_include_directories(flexitag_py PRIVATE ${ICU_INCLUDE_DIR})
        endif()
    endif()
endif()

