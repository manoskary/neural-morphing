cmake_minimum_required(VERSION 3.22)
project(NeuralMorphing VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Expect JUCE to be present as a subdirectory. You can alternatively fetch it via CPM.
add_subdirectory(JUCE)

set(SOURCE_FILES
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h
    Source/LockFreeRing.h
    Source/ModelBackend.cpp
    Source/ModelBackend.h
    Source/PaletteIndex.cpp
    Source/PaletteIndex.h
    Source/OnsetDetector.cpp
    Source/OnsetDetector.h
    Source/Workers.cpp
    Source/Workers.h
)

juce_add_binary_data(NeuralMorphingBinaryData
    SOURCES
        Resources/knob1.png
        Resources/name_long_logo.png
        Resources/vst_background.png
)

juce_add_plugin(NeuralMorphing
    COMPANY_NAME "NeuralMorph"
    BUNDLE_ID com.neuralmorph.plugin
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/Resources/app_icon.png"
    ICON_SMALL "${CMAKE_CURRENT_SOURCE_DIR}/Resources/app_icon.png"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    FORMATS VST3 Standalone
    PRODUCT_NAME "Neural Morphing"
)

target_sources(NeuralMorphing PRIVATE ${SOURCE_FILES})

target_link_libraries(NeuralMorphing PRIVATE NeuralMorphingBinaryData)

target_compile_definitions(NeuralMorphing
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_IGNORE_VST3_MISMATCHED_PARAMETER_ID_WARNING=1
)

if (WIN32)
    target_compile_definitions(NeuralMorphing PUBLIC JUCE_USE_CURL=0)
else()
    target_compile_definitions(NeuralMorphing PUBLIC JUCE_USE_CURL=1)
endif()

juce_generate_juce_header(NeuralMorphing)

target_link_libraries(NeuralMorphing PRIVATE
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_gui_basics
    juce::juce_gui_extra
    juce::juce_data_structures
)

option(NEURAL_MORPHING_ENABLE_ONNX "Build with ONNX Runtime backend" ON)
option(NM_WITH_PYBRIDGE "Build with Python bridge HTTP backend" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime SDK root containing include/ and lib/")
if (NEURAL_MORPHING_ENABLE_ONNX)
    find_package(onnxruntime QUIET)
    if (NOT onnxruntime_FOUND AND ONNXRUNTIME_ROOT)
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            PATHS "${ONNXRUNTIME_ROOT}/include"
            NO_DEFAULT_PATH)
        find_library(ONNXRUNTIME_LIBRARY onnxruntime
            PATHS "${ONNXRUNTIME_ROOT}/lib"
            NO_DEFAULT_PATH)
        find_file(ONNXRUNTIME_DLL onnxruntime.dll
            PATHS "${ONNXRUNTIME_ROOT}/lib"
            NO_DEFAULT_PATH)
        if (ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
            add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
            set(onnxruntime_FOUND TRUE)
        endif()
    endif()
    if (onnxruntime_FOUND)
        target_compile_definitions(NeuralMorphing PRIVATE NM_HAS_ONNX=1)
        target_link_libraries(NeuralMorphing PRIVATE onnxruntime::onnxruntime)
        target_sources(NeuralMorphing PRIVATE
            Source/ModelBackendOnnx.cpp
            Source/ModelBackendOnnx.h
        )
        message(STATUS "ONNX Runtime backend enabled")
        if (MSVC)
            foreach(_nm_wrapper_target NeuralMorphing_VST3 NeuralMorphing_Standalone)
                target_link_options(${_nm_wrapper_target} PRIVATE "/DELAYLOAD:onnxruntime.dll")
                target_link_libraries(${_nm_wrapper_target} PRIVATE delayimp)
            endforeach()
        endif()
        if (ONNXRUNTIME_DLL)
            foreach(_nm_wrapper_target NeuralMorphing_VST3 NeuralMorphing_Standalone)
                if (TARGET ${_nm_wrapper_target})
                    add_custom_command(TARGET ${_nm_wrapper_target} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                "${ONNXRUNTIME_DLL}"
                                "$<TARGET_FILE_DIR:${_nm_wrapper_target}>")
                endif()
            endforeach()
        endif()
    else()
        target_compile_definitions(NeuralMorphing PRIVATE NM_HAS_ONNX=0)
        message(WARNING "ONNX Runtime not found; building with stub backend only")
    endif()
else()
    target_compile_definitions(NeuralMorphing PRIVATE NM_HAS_ONNX=0)
endif()

# Python Bridge HTTP backend
if (NM_WITH_PYBRIDGE)
    target_compile_definitions(NeuralMorphing PRIVATE NM_WITH_PYBRIDGE=1)
    target_sources(NeuralMorphing PRIVATE
        Source/ModelBackendHttp.cpp
        Source/ModelBackendHttp.h
    )
    if (NOT WIN32)
        find_package(CURL REQUIRED)
        target_link_libraries(NeuralMorphing PRIVATE CURL::libcurl)
    endif()
    message(STATUS "Python bridge HTTP backend enabled")
else()
    target_compile_definitions(NeuralMorphing PRIVATE NM_WITH_PYBRIDGE=0)
    message(STATUS "Python bridge HTTP backend disabled")
endif()

# Installation helpers: copy the built VST3 to typical locations.
if (APPLE)
    set(_vst_dest "$ENV{HOME}/Library/Audio/Plug-Ins/VST3")
    install(DIRECTORY "${CMAKE_BINARY_DIR}/NeuralMorphing_artefacts/VST3/" DESTINATION "${_vst_dest}" USE_SOURCE_PERMISSIONS)
elseif (WIN32)
    set(_vst_dest "$ENV{APPDATA}/VST3")
    install(DIRECTORY "${CMAKE_BINARY_DIR}/NeuralMorphing_artefacts/VST3/" DESTINATION "${_vst_dest}" USE_SOURCE_PERMISSIONS)
endif()
