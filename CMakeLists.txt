message("Platform: " ${CMAKE_SYSTEM_NAME})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(ONNX_PATH ${CMAKE_CURRENT_LIST_DIR}/lib/aarch64)
        message("Compiling for Linux aarch64")
        FetchContent_Declare(
            onnxruntime
            URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-aarch64-1.18.0.tgz
            SOURCE_DIR ${ONNX_PATH}
        )
        find_package(SDL2 REQUIRED)
        include_directories(${SDL2_INCLUDE_DIRS})
        FetchContent_MakeAvailable(onnxruntime)
        add_library(onnxruntime SHARED IMPORTED )
        # You can define two import-locations: one for debug and one for release.
        set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNX_PATH}/lib/libonnxruntime.so)
        add_library(openwakeword ${CMAKE_CURRENT_LIST_DIR}/src/openwakeword.cpp)
        target_include_directories(openwakeword PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/ ${ONNX_PATH}/include ${SDL2_INCLUDE_DIRS})
        target_link_libraries(openwakeword PUBLIC onnxruntime ${SDL2_LIBRARIES})
        add_compile_definitions(openwakeword PRIVATE -DMODEL_PATH=${CMAKE_CURRENT_LIST_DIR}/model)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(ONNX_PATH ${CMAKE_CURRENT_LIST_DIR}/lib/x64)
        message("Compiling for Linux x86_64")
        include(FetchContent)
        FetchContent_Declare(
            onnxruntime
            URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
            SOURCE_DIR ${ONNX_PATH}
        )
        find_package(SDL2 REQUIRED)
        include_directories(${SDL2_INCLUDE_DIRS})
        
        FetchContent_MakeAvailable(onnxruntime)
        add_library(onnxruntime SHARED IMPORTED )
        # You can define two import-locations: one for debug and one for release.
        set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNX_PATH}/lib/libonnxruntime.so)
        add_library(openwakeword ${CMAKE_CURRENT_LIST_DIR}/src/openwakeword.cpp)
        target_include_directories(openwakeword PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/ ${ONNX_PATH}/include ${SDL2_INCLUDE_DIRS})
        target_link_libraries(openwakeword PUBLIC onnxruntime ${SDL2_LIBRARIES})
        add_compile_definitions(openwakeword PRIVATE -DMODEL_PATH="${CMAKE_CURRENT_LIST_DIR}/model")
    endif()
endif()
