cmake_minimum_required(VERSION 3.10)

project(IBloFunMatch VERSION 0.0)

add_executable(IBloFunMatch IBloFunMatch.cc)

target_include_directories(IBloFunMatch PRIVATE "ext/")

find_package(Boost REQUIRED COMPONENTS program_options)

target_link_libraries(IBloFunMatch PRIVATE Boost::program_options)