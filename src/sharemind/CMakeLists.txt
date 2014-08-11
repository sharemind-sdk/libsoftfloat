#
# This file is a part of the Sharemind framework.
# Copyright (C) Cybernetica AS
#
# All rights are reserved. Reproduction in whole or part is prohibited
# without the written consent of the copyright owner. The usage of this
# code is subject to the appropriate license agreement.
#

CMAKE_MINIMUM_REQUIRED(VERSION "2.8.12")
CMAKE_POLICY(SET CMP0015 NEW)
PROJECT("SHAREMIND_LIBSOFTFLOAT" "C")

INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/config.local" OPTIONAL)
INCLUDE("${CMAKE_CURRENT_BINARY_DIR}/config.local" OPTIONAL)

FOREACH(dir IN LISTS CMAKE_PREFIX_PATH)
  INCLUDE_DIRECTORIES(BEFORE SYSTEM "${dir}/include")
ENDFOREACH()
INCLUDE_DIRECTORIES(BEFORE SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/../")

ADD_COMPILE_OPTIONS(
  "-O2" "-Wextra" "-Wall"
  "$<$<NOT:$<STREQUAL:$<CONFIGURATION>,Release>>:-ggdb>"
  "$<$<NOT:$<STREQUAL:$<CONFIGURATION>,Release>>:-fno-omit-frame-pointer>"
  "$<$<STREQUAL:$<CONFIGURATION>,Release>:-DNDEBUG>"
  "$<$<STREQUAL:$<CONFIGURATION>,Release>:-fomit-frame-pointer>"
)

SET(CMAKE_C_FLAGS "-std=c99 -Wall -Wextra -fstrict-aliasing -Wstrict-aliasing=1 -fwrapv -Wabi -Wbad-function-cast -Wc++-compat -Wformat -Wswitch-default -Wunused -Wfloat-equal -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wconversion -Wsign-conversion -Wunused-macros")

INCLUDE(CheckCCompilerFlag)
MACRO(FLAGS_MAGIC flags)
    CHECK_C_COMPILER_FLAG("-${flags}" SHAREMIND_HAS_CFLAGS__${flags})
    IF(SHAREMIND_HAS_CFLAGS__${flags})
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -${flags}")
    ENDIF(SHAREMIND_HAS_CFLAGS__${flags})
ENDMACRO(FLAGS_MAGIC)
FLAGS_MAGIC("Weverything")
FLAGS_MAGIC("Wlogical-op")

SET(LIBSOFTFLOAT_INSTALL_DEFINITIONS "-D__STDC_LIMIT_MACROS")
ADD_DEFINITIONS("${LIBSOFTFLOAT_INSTALL_DEFINITIONS}")

################################################################################
# libsoftfloat
################################################################################

SET(SHAREMIND_LIBSOFTFLOAT_HEADERS
    "3rdparty/libsoftfloat/milieu.h"
    "3rdparty/libsoftfloat/softfloat.h"
)

ADD_LIBRARY("libsoftfloat" STATIC "3rdparty/libsoftfloat/softfloat.c" ${SHAREMIND_LIBSOFTFLOAT_HEADERS})
SET_SOURCE_FILES_PROPERTIES("3rdparty/libsoftfloat/softfloat.c" PROPERTIES
                            COMPILE_FLAGS "-fPIC -ggdb")

SET(LIBSOFTFLOAT_VERSION "0.1.0.0")
SET_TARGET_PROPERTIES("libsoftfloat" PROPERTIES
                      OUTPUT_NAME "sharemind_softfloat"
                      VERSION "${LIBSOFTFLOAT_VERSION}"
                      SOVERSION "0.1"
                      COMPILE_FLAGS "-frounding-math -ftrapping-math -fsignaling-nans")

INSTALL(TARGETS "libsoftfloat"
        ARCHIVE DESTINATION "lib")

INSTALL(FILES ${SHAREMIND_LIBSOFTFLOAT_HEADERS}
        DESTINATION "include/sharemind/3rdparty/libsoftfloat/")

# Install cmake files:
FILE(GLOB_RECURSE sharemind_libsoftfloat_in_files "${CMAKE_CURRENT_SOURCE_DIR}/*.in")
ADD_CUSTOM_TARGET("include_sharemind_libsoftfloat_in_files_in_IDE" SOURCES ${sharemind_libsoftfloat_in_files})
SET(LIBSOFTFLOAT_INSTALL_LINK_LIBRARIES "-L${CMAKE_INSTALL_PREFIX}/lib" "-lsharemind_softfloat")
SET(LIBSOFTFLOAT_INSTALL_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")

CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/SharemindLibsoftfloatConfig.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/SharemindLibsoftfloatConfig.cmake" @ONLY)
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/SharemindLibsoftfloatConfigVersion.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/SharemindLibsoftfloatConfigVersion.cmake" @ONLY)
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/SharemindLibsoftfloatConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/SharemindLibsoftfloatConfigVersion.cmake"
        DESTINATION "lib/SharemindLibsoftfloat")
