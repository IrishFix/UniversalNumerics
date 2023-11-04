cmake_minimum_required(VERSION 3.10)

project(UniversalNumericsTests)

add_executable(TestExecutable main.cpp)

target_link_libraries(TestExecutable UniversalNumerics)

target_include_directories(TestExecutable PUBLIC ../include)