cmake_minimum_required(VERSION 3.10)

project(IBloFunMatch_p 
		VERSION 0.0
		LANGUAGES CXX)

add_executable(IBloFunMatch IBloFunMatch.cc)

target_include_directories(IBloFunMatch PRIVATE "ext/")

target_include_directories(IBloFunMatch PRIVATE "sorting/")

find_package(Boost COMPONENTS program_options REQUIRED)

target_link_libraries(IBloFunMatch PRIVATE Boost::program_options)



# Save executable path into file
file(WRITE exe_path.txt ${CMAKE_BINARY_DIR})
file(APPEND exe_path.txt "/IBloFunMatch")