cmake_minimum_required(VERSION 3.10)

project(IBloFunMatch_p 
		VERSION 0.0
		LANGUAGES CXX)

option(DEBUG_MERGES "Print matrices usied when computing merge distances" OFF)

add_executable(IBloFunMatch IBloFunMatch.cc)

target_include_directories(IBloFunMatch PRIVATE "ext/")

target_include_directories(IBloFunMatch PRIVATE "sorting/")

find_package(Boost COMPONENTS program_options REQUIRED)

target_link_libraries(IBloFunMatch PRIVATE Boost::program_options)

if(DEBUG_MERGES)
	target_compile_definitions(IBloFunMatch DEBUG_MERGES)
endif()


# Save executable path into file
file(WRITE exe_path.txt ${CMAKE_BINARY_DIR})
file(APPEND exe_path.txt "/IBloFunMatch")