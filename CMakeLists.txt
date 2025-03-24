# installed proto to /usr/local/include
cmake_minimum_required(VERSION 3.16)
project(my_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    /usr/include/
    /usr/local/include/OpenCNN/include/
    /home/kek/Documents/rudens/praktika/prof_praktika/network/dynet/

)


add_executable(my_project main.cpp cnn.cpp wav.cpp transform_data.cpp segment_data.cpp)

link_directories(/usr/lib)

target_link_libraries(my_project PRIVATE 
    fftw3
    glog
    /home/kek/Documents/rudens/praktika/prof_praktika/network/dynet/build/dynet/libdynet.so
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_STRONG_INLINE=inline -Wno-sign-compare -O2 -fPIC")
