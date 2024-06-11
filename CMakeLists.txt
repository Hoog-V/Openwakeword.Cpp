message("Platform: " ${CMAKE_SYSTEM_NAME})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        message("Compiling for Linux aarch64")
        FetchContent_Declare(
            onnxruntime
            URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-aarch64-1.18.0.tgz
            SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/aarch64
        )
        FetchContent_MakeAvailable(onnxruntime)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(ONNX_PATH ${CMAKE_CURRENT_LIST_DIR}/lib/x64)
        message("Compiling for Linux x86_64")
        include(FetchContent)
        FetchContent_Declare(
            onnxruntime
            URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
            SOURCE_DIR ${ONNX_PATH}
        )
        FetchContent_MakeAvailable(onnxruntime)
        add_library(onnxruntime SHARED IMPORTED )
        # You can define two import-locations: one for debug and one for release.
        set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNX_PATH}/lib/libonnxruntime.so)
        add_library(openwakeword INTERFACE ${ONNX_PATH}/lib/libonnxruntime.so ${CMAKE_CURRENT_LIST_DIR}/src/openwakeword.cpp)
        target_include_directories(openwakeword INTERFACE ${CMAKE_CURRENT_LIST_DIR}/src/ ${ONNX_PATH}/include)
        target_link_libraries(openwakeword INTERFACE onnxruntime)
    endif()
endif()