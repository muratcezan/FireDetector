cmake_minimum_required(VERSION 3.10)

# Proje adı
project(FireDetection)

# C++ standardını belirliyoruz
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV paketlerini buluyoruz
find_package(OpenCV REQUIRED)

# Kaynak dosyayı ekliyoruz
add_executable(FireDetection main.cpp)

# OpenCV kütüphanelerini projemize bağlıyoruz
target_link_libraries(FireDetection ${OpenCV_LIBS})
