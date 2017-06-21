#
# Copyright (c) 2015, Cybernetica AS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#   * Neither the name of Cybernetica AS nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL CYBERNETICA AS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

CMAKE_MINIMUM_REQUIRED(VERSION "2.8.12")
PROJECT("SHAREMIND_LIBSOFTFLOAT" "C")

SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "1")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(LIBSOFTFLOAT_SOVERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}")
SET(LIBSOFTFLOAT_VERSION
    "${LIBSOFTFLOAT_SOVERSION}.${CPACK_PACKAGE_VERSION_PATCH}")

INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/config.local" OPTIONAL)
INCLUDE("${CMAKE_CURRENT_BINARY_DIR}/config.local" OPTIONAL)

ADD_COMPILE_OPTIONS(
  "-std=c99" "-O2" "-fstrict-aliasing" "-fwrapv" "-fPIC" "-Wextra" "-Wall"
  "-Wstrict-aliasing=1" "-Wabi" "-Wbad-function-cast" "-Wc++-compat" "-Wformat"
  "-Wswitch-default" "-Wunused" "-Wfloat-equal" "-Wshadow" "-Wpointer-arith"
  "-Wcast-qual" "-Wstrict-prototypes" "-Wconversion" "-Wsign-conversion"
  "-Wunused-macros"
  "$<$<NOT:$<STREQUAL:$<CONFIGURATION>,Release>>:-ggdb>"
  "$<$<NOT:$<STREQUAL:$<CONFIGURATION>,Release>>:-fno-omit-frame-pointer>"
  "$<$<STREQUAL:$<CONFIGURATION>,Release>:-DNDEBUG>"
  "$<$<STREQUAL:$<CONFIGURATION>,Release>:-fomit-frame-pointer>"
)

INCLUDE(CheckCCompilerFlag)
MACRO(FLAGS_MAGIC flags)
    CHECK_C_COMPILER_FLAG("-${flags}" SHAREMIND_HAS_CFLAGS__${flags})
    IF(SHAREMIND_HAS_CFLAGS__${flags})
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -${flags}")
    ENDIF(SHAREMIND_HAS_CFLAGS__${flags})
ENDMACRO(FLAGS_MAGIC)
FLAGS_MAGIC("Weverything")
FLAGS_MAGIC("Wlogical-op")
FLAGS_MAGIC("Wno-padded")

SET(LIBSOFTFLOAT_INSTALL_DEFINITIONS
    "-D__STDC_LIMIT_MACROS"
    "-D__STDC_CONSTANT_MACROS")
ADD_DEFINITIONS(${LIBSOFTFLOAT_INSTALL_DEFINITIONS})

################################################################################
# libsoftfloat
################################################################################

SET(SHAREMIND_LIBSOFTFLOAT_HEADERS
    "3rdparty/libsoftfloat/macros.h"
    "3rdparty/libsoftfloat/milieu.h"
    "3rdparty/libsoftfloat/softfloat.h"
    "3rdparty/libsoftfloat/specialize.h"
)

ADD_LIBRARY("libsoftfloat" STATIC "3rdparty/libsoftfloat/softfloat.c"
                                  ${SHAREMIND_LIBSOFTFLOAT_HEADERS})

SET_TARGET_PROPERTIES("libsoftfloat" PROPERTIES
                      OUTPUT_NAME "sharemind_softfloat"
                      VERSION "${LIBSOFTFLOAT_VERSION}"
                      SOVERSION "${LIBSOFTFLOAT_SOVERSION}")

INSTALL(TARGETS "libsoftfloat"
        COMPONENT static
        ARCHIVE DESTINATION "lib")

INSTALL(FILES ${SHAREMIND_LIBSOFTFLOAT_HEADERS}
        DESTINATION "include/sharemind/3rdparty/libsoftfloat/"
        COMPONENT dev)

# Install cmake files:
FILE(GLOB_RECURSE sharemind_libsoftfloat_in_files
     "${CMAKE_CURRENT_SOURCE_DIR}/*.in")
ADD_CUSTOM_TARGET("include_sharemind_libsoftfloat_in_files_in_IDE" SOURCES
                  ${sharemind_libsoftfloat_in_files})
SET(LIBSOFTFLOAT_INSTALL_LINK_LIBRARIES
    "-L${CMAKE_INSTALL_PREFIX}/lib"
    "-lsharemind_softfloat"
)

SET(LIBSOFTFLOAT_INSTALL_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")

CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/SharemindLibsoftfloatConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SharemindLibsoftfloatConfig.cmake" @ONLY)
CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/SharemindLibsoftfloatConfigVersion.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SharemindLibsoftfloatConfigVersion.cmake"
    @ONLY)
INSTALL(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SharemindLibsoftfloatConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SharemindLibsoftfloatConfigVersion.cmake"
    DESTINATION "lib/SharemindLibsoftfloat"
    COMPONENT dev)


# Packaging:
SET(CPACK_PACKAGE_VENDOR "Cybernetica AS")
SET(CPACK_PACKAGE_CONTACT "Sharemind packaging <sharemind-packaging@cyber.ee>")
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_NAME "libsharemind-softfloat")
SET(CPACK_DEBIAN_PACKAGE_RELEASE "cyber1")
SET(CPACK_DEBIAN_COMPRESSION_TYPE "xz")
SET(CPACK_DEB_COMPONENT_INSTALL "ON")
SET(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

SET(CPACK_COMPONENT_DEV_DESCRIPTION
    "Sharemind SoftFloat library development headers")
SET(CPACK_COMPONENT_STATIC_DESCRIPTION "Sharemind SoftFloat library (static)")
SET(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS
    "libsharemind-softfloat-static (= ${LIBSOFTFLOAT_VERSION}), libc6-dev (>= 2.19)")
SET(CPACK_DEBIAN_STATIC_PACKAGE_DEPENDS "libc6 (>= 2.19)")
SET(CPACK_DEBIAN_DEV_PACKAGE_SECTION "libdevel")
SET(CPACK_DEBIAN_STATIC_PACKAGE_SECTION "libs")
SET(CPACK_DEBIAN_DEV_PACKAGE_ARCHITECTURE "any")
INCLUDE(CPack)
