find_library(MATH_LIBRARY m)

# check systems
if (NOT UNAME_S)
    execute_process(COMMAND uname -s OUTPUT_VARIABLE UNAME_S)
endif()
if (NOT UNAME_P)
    execute_process(COMMAND uname -p OUTPUT_VARIABLE UNAME_P)
endif()
if (NOT UNAME_M)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE UNAME_M)
endif()
#message(STATUS "UNAME_S: ${UNAME_S}  UNAME_P: ${UNAME_P}  UNAME_M: ${UNAME_M}")

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
if (UNAME_S MATCHES "Darwin")
    if (NOT UNAME_P MATCHES "arm")
        execute_process(COMMAND sysctl -n hw.optional.arm64 OUTPUT_VARIABLE SYSCTL_M)
        if (SYSCTL_M MATCHES "1")
            #set(UNAME_P "arm")
            #set(UNAME_M "arm64")
            message(WARNING "Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lea
d to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-#1282546789")
        endif()
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(aarch64|arm.*|ARM64)$")
    message(STATUS "ARM detected")
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=apple-m1")
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PPC64 detected")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mpower9-vector")
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    message(STATUS "loongarch64 detected")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mlsx -mlasx")
else()
    message(STATUS "x86 detected")
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2 -mfma -mf16c")
    if (UNAME_S MATCHES "Darwin")
        execute_process(COMMAND sysctl machdep.cpu.features OUTPUT_VARIABLE AVX1_M)
        if (AVX1_M MATCHES "AVX1.0")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
        endif()
        execute_process(COMMAND sysctl machdep.cpu.leaf7_features OUTPUT_VARIABLE AVX2_M)
        if (AVX2_M MATCHES "AVX2")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
        endif()
        if (AVX1_M MATCHES "FMA")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
        endif()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
    elseif (UNAME_S MATCHES "Linux")
        message(STATUS "Linux detected")
        # must have to build on ubuntu22 with gcc11:
        find_package(Threads)
        set(GGML_EXTRA_LIBS  ${GGML_EXTRA_LIBS} Threads::Threads)

        execute_process(COMMAND grep "avx " /proc/cpuinfo OUTPUT_VARIABLE AVX1_M)
        if (AVX1_M MATCHES "avx")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
        endif()
        execute_process(COMMAND grep "avx2 " /proc/cpuinfo OUTPUT_VARIABLE AVX2_M)
        if (AVX2_M MATCHES "avx2")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
        endif()
        execute_process(COMMAND grep "fma " /proc/cpuinfo OUTPUT_VARIABLE FMA_M)
        if (FMA_M MATCHES "fma")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
        endif()
        execute_process(COMMAND grep "f16c " /proc/cpuinfo OUTPUT_VARIABLE F16C_M)
        if (F16C_M MATCHES "f16c")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
        endif()
        execute_process(COMMAND grep "sse3 " /proc/cpuinfo OUTPUT_VARIABLE SSE3_M)
        if (SSE3_M MATCHES "sse3")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
        endif()
    elseif (UNAME_S MATCHES "Haiku")
        message(STATUS "Haiku detected")
	execute_process(COMMAND sysinfo -cpu COMMAND grep "AVX " OUTPUT_VARIABLE AVX1_M)
        if (AVX1_M MATCHES "avx")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
        endif()
	execute_process(COMMAND sysinfo -cpu COMMAND grep "AVX2 " OUTPUT_VARIABLE AVX2_M)
        if (AVX2_M MATCHES "avx2")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
        endif()
	execute_process(COMMAND sysinfo -cpu COMMAND grep "FMA " OUTPUT_VARIABLE FMA_M)
        if (FMA_M MATCHES "fma")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
        endif()
	execute_process(COMMAND sysinfo -cpu COMMAND grep "F16C " OUTPUT_VARIABLE F16C_M)
        if (F16C_M MATCHES "f16c")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
        endif()
    elseif (MSVC)
        if (GGML_AVX512)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX512")
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (GGML_AVX512_VBMI)
                add_compile_definitions(__AVX512VBMI__)
            endif()
            if (GGML_AVX512_VNNI)
                add_compile_definitions(__AVX512VNNI__)
            endif()
        elseif (GGML_AVX2)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
        elseif (GGML_AVX)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX")
        endif()
    else()
        set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -mfma -mf16c -mavx -mavx2")
    endif()
endif()

# on APPLE - include Accelerate framework
if (APPLE AND NOT GGML_NO_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        set(GGML_EXTRA_LIBS  ${GGML_EXTRA_LIBS}  ${ACCELERATE_FRAMEWORK})
        set(GGML_EXTRA_FLAGS ${GGML_EXTRA_FLAGS} -DGGML_USE_ACCELERATE)
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (GGML_OPENBLAS)
    set(OPENBLAS_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/include/openblas
        /usr/include/openblas-base
        /usr/local/include
        /usr/local/include/openblas
        /usr/local/include/openblas-base
        /opt/OpenBLAS/include
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/include
        )
    find_path(OPENBLAS_INC NAMES cblas.h PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS})
    find_library(OPENBLAS_LIB NAMES openblas libopenblas)
    if (OPENBLAS_LIB)
        message(STATUS "OpenBLAS found")

        set(GGML_EXTRA_LIBS  ${GGML_EXTRA_LIBS}  ${OPENBLAS_LIB})
        set(GGML_EXTRA_INCS  ${GGML_EXTRA_INCS}  ${OPENBLAS_INC})
        set(GGML_EXTRA_FLAGS ${GGML_EXTRA_FLAGS} -DGGML_USE_OPENBLAS)
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

# undefine NDEBUG so asserts don't get disabled in tests
add_definitions(-UNDEBUG)

#
# The client links only core ggml, including with GGML_BACKEND_DL=ON.
set(TEST_TARGET test-cpu-graph-plan)
add_executable(${TEST_TARGET} test-cpu-graph-plan.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-backend-buft-queries)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-opencl-kernel-sources)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}> ${CMAKE_CURRENT_SOURCE_DIR}/../src/ggml-opencl/kernels)

set(TEST_TARGET test-backend-memory)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

# test-backend-ops

set(TEST_TARGET test-backend-ops)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml Threads::Threads)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

# Backends whose device memory shares the host's physical pool must report it
# through ggml_backend_dev_props.memory_unified.
set(TEST_TARGET test-unified-memory-props)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
if (GGML_METAL)
    target_compile_definitions(${TEST_TARGET} PRIVATE GGML_TEST_EXPECT_METAL)
endif()
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

# Installed-package consumer: find_package(ggml) with ggml::all (and the vector
# index when built), plus the pkg-config metadata against the installed libdir.
if (NOT GGML_BACKEND_DL OR GGML_CPU_STATIC)
    set(GGML_CONSUMER_EXPECT_CPU ON)
else()
    set(GGML_CONSUMER_EXPECT_CPU OFF)
endif()
function(ggml_collect_module_backends out_var)
    set(module_backends "")
    foreach(backend IN LISTS GGML_AVAILABLE_BACKENDS)
        if (TARGET "${backend}")
            get_target_property(backend_type "${backend}" TYPE)
            if (backend_type STREQUAL "MODULE_LIBRARY")
                list(APPEND module_backends "${backend}")
            endif()
        endif()
    endforeach()
    string(JOIN "," joined ${module_backends})
    set(${out_var} "${joined}" PARENT_SCOPE)
endfunction()
ggml_collect_module_backends(GGML_CONSUMER_MODULE_BACKENDS)
add_test(NAME test-package-consumer
    COMMAND ${CMAKE_COMMAND}
        -DGGML_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DGGML_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/package-consumer
        -DGGML_LIB_OUTPUT_PREFIX=${GGML_LIB_OUTPUT_PREFIX}
        -DGGML_CONSUMER_EXPECT_CPU=${GGML_CONSUMER_EXPECT_CPU}
        -DGGML_CONSUMER_MODULE_BACKENDS=${GGML_CONSUMER_MODULE_BACKENDS}
        -DGGML_CONSUMER_CONFIG=$<CONFIG>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/package-consumer/run.cmake)

# ggml-opencl flash-attention q-chunking parity against the CPU backend.
# Self-skipping at runtime when no OpenCL device is present.
set(TEST_TARGET test-opencl-fa-chunking)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

if (TARGET ggml-vector-index)
    set(TEST_TARGET test-vector-index)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml-vector-index)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

    set(TEST_TARGET test-vector-index-hooks)
    add_executable(${TEST_TARGET} test-vector-index.cpp)
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_compile_definitions(${TEST_TARGET} PRIVATE GGML_VEC_INDEX_TEST_HOOKS)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml-vector-index-test)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}> --internal-hooks)

    set(TEST_TARGET test-vector-index-faults)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml-vector-index-test)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endif()

#
# test-adreno-version
# Pure unit test for src/ggml-adreno.h (Adreno GPU-string parser used by the
# Android OpenCL/Vulkan backend-selection policy). Header-only: needs the src/
# include path but does not link the ggml runtime. Always built.

set(TEST_TARGET test-adreno-version)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

set(TEST_TARGET test-shader-payload-strip)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/ggml-vulkan/vulkan-shaders)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

if (GGML_OPENCL)
    set(TEST_TARGET test-opencl-workgroup)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endif()

if (GGML_METAL)
    set(TEST_TARGET test-metal-free-memory)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endif()

if (GGML_VULKAN)
    set(TEST_TARGET test-vulkan-status)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    # ggml-vulkan exports the testing hook; ggml carries the public headers.
    target_link_libraries(${TEST_TARGET} PRIVATE ggml-vulkan ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endif()

if (NOT GGML_BACKEND_DL)
    #
    # test-opt
    set(TEST_TARGET test-opt)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-quantize-fns

    set(TEST_TARGET test-quantize-fns)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-quantize-perf

    set(TEST_TARGET test-quantize-perf)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-pool

    set(TEST_TARGET test-pool)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    if (MSVC)
        target_link_options(${TEST_TARGET} PRIVATE "/STACK:8388608") # 8MB
    endif()
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-arange

    set(TEST_TARGET test-arange)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml Threads::Threads)
    if (MSVC)
        target_link_options(${TEST_TARGET} PRIVATE "/STACK:8388608") # 8MB
    endif()
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-timestep_embedding

    set(TEST_TARGET test-timestep_embedding)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    if (MSVC)
        target_link_options(${TEST_TARGET} PRIVATE "/STACK:8388608") # 8MB
    endif()
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-pad-reflect-1d

    set(TEST_TARGET test-pad-reflect-1d)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

    #
    # test-roll

    set(TEST_TARGET test-roll)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

    #
    # test-col2im-1d

    set(TEST_TARGET test-col2im-1d)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-snake

    set(TEST_TARGET test-snake)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-vae-graph

    set(TEST_TARGET test-vae-graph)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # vae-decode (tool: loads real acestep VAE GGUF -> WAV; no add_test, needs weights)

    set(TEST_TARGET vae-decode)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)

    #
    # vae-roundtrip (tool: WAV -> encode -> decode -> WAV; no add_test, needs weights)

    set(TEST_TARGET vae-roundtrip)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)

    #
    # test-conv-transpose

    set(TEST_TARGET test-conv-transpose)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

    # test-conv-transpose-1d

    set(TEST_TARGET test-conv-transpose-1d)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

    #
    # test-dup

    set(TEST_TARGET test-dup)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

    #
    # test-rel-pos

    set(TEST_TARGET test-rel-pos)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

    #
    # test-customop

    set(TEST_TARGET test-customop)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    if (MSVC)
        target_link_options(${TEST_TARGET} PRIVATE "/STACK:8388608") # 8MB
    endif()
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-conv1d

    set(TEST_TARGET test-conv1d)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-conv1d-dw-c1

    set(TEST_TARGET test-conv1d-dw-c1)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-conv1d-dw-c2

    set(TEST_TARGET test-conv1d-dw-c2)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-conv2d

    set(TEST_TARGET test-conv2d)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-conv2d-dw

    set(TEST_TARGET test-conv2d-dw)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-cont

    set(TEST_TARGET test-cont)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.c)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")

    #
    # test-interpolate

    set(TEST_TARGET test-interpolate)
    add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY ENVIRONMENT "LLVM_PROFILE_FILE=${TEST_TARGET}.profraw")
endif()

set(TEST_TARGET test-conv-transpose-1d-scaling)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

set(TEST_TARGET test-hexagon-buffer)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

set(TEST_TARGET test-hexagon-remap)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

set(TEST_TARGET test-hexagon-mul-mat-q8)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

# Host-portable regression for metadata retained until a Hexagon queue reply.
set(TEST_TARGET test-hexagon-opcache)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_include_directories(${TEST_TARGET} PRIVATE
    ../src ../src/ggml-hexagon ../src/ggml-hexagon/htp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-hexagon-runtime-info)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_include_directories(${TEST_TARGET} PRIVATE
    ../src ../src/ggml-hexagon ../src/ggml-hexagon/htp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)

set(TEST_TARGET test-hexagon-runtime-info-device)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

set(TEST_TARGET test-audio8-codec-f32)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_tests_properties(${TEST_TARGET} PROPERTIES SKIP_RETURN_CODE 77
    ENVIRONMENT "GGML_HEXAGON_MM_SELECT=2;GGML_HEXAGON_OPSTAGE=3")
add_test(NAME ${TEST_TARGET}-reference COMMAND $<TARGET_FILE:${TEST_TARGET}> --reference-only)

set(TEST_TARGET test-hexagon-conv-taps)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_tests_properties(${TEST_TARGET} PROPERTIES SKIP_RETURN_CODE 77
    ENVIRONMENT "GGML_HEXAGON_OPSTAGE=3;GGML_HEXAGON_F16_F32_BLOCKED=0;GGML_HEXAGON_F16_F32_FUSE_ADD=0;GGML_HEXAGON_F16_F32_PANEL=1;GGML_HEXAGON_F16_F32_PANEL_SHAPE=4x2")
