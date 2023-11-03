project(UniversalNumerics)

cmake_minimum_required(VERSION 3.10)

set(SOURCES
        Networking/MLP.cpp
        # ... other source files ...
)

add_library(UniversalNumerics SHARED ${SOURCES})

target_include_directories(UniversalNumerics PUBLIC ../include)

set_target_properties(UniversalNumerics PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)