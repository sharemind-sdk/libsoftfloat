#
# Copyright (c) Cybernetica AS
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

CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
PROJECT(SharemindLibSoftfloat VERSION 0.3.0 LANGUAGES C)

INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/config.local" OPTIONAL)
INCLUDE("${CMAKE_CURRENT_BINARY_DIR}/config.local" OPTIONAL)

FIND_PACKAGE(SharemindCMakeHelpers 1.6 REQUIRED)

FIND_PACKAGE(SharemindCHeaders 1.3.0 REQUIRED)


# LibSoftfloat:
FILE(GLOB SharemindLibSoftfloat_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
FILE(GLOB SharemindLibSoftfloat_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
SharemindAddSharedLibrary(LibSoftfloat
    OUTPUT_NAME "sharemind_softfloat"
    SOURCES
        ${SharemindLibSoftfloat_HEADERS}
        ${SharemindLibSoftfloat_SOURCES}
)
TARGET_INCLUDE_DIRECTORIES(LibSoftfloat
    INTERFACE
        # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> # TODO
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
    )
TARGET_COMPILE_OPTIONS(LibSoftfloat PRIVATE "-fwrapv") # TODO: Is this needed?
TARGET_COMPILE_DEFINITIONS(LibSoftfloat
    PUBLIC
        "__STDC_LIMIT_MACROS"
        "__STDC_CONSTANT_MACROS"
    )
TARGET_LINK_LIBRARIES(LibSoftfloat PUBLIC Sharemind::CHeaders)
INSTALL(FILES ${SharemindLibSoftfloat_HEADERS}
        DESTINATION "include/sharemind/libsoftfloat/"
        COMPONENT "dev")
SharemindCreateCMakeFindFilesForTarget(LibSoftfloat
    DEPENDENCIES
        "SharemindCHeaders 1.3.0"
)


# Packaging:
SharemindSetupPackaging()
SharemindAddComponentPackage("lib"
    NAME "libsharemind-softfloat"
    DESCRIPTION "Sharemind SoftFloat library"
    DEB_SECTION "libs"
    DEB_DEPENDS "libc6 (>= 2.19)")
SharemindAddComponentPackage("dev"
    NAME "libsharemind-softfloat-dev"
    DESCRIPTION "Sharemind SoftFloat library development headers"
    DEB_SECTION "libdevel"
    DEB_DEPENDS
        "libsharemind-softfloat (= ${SharemindLibSoftfloat_DEB_lib_PACKAGE_VERSION})"
        "libsharemind-cheaders-dev (>= 1.3.0)"
        "libc6-dev (>= 2.19)"
)
SharemindAddComponentPackage("debug"
    NAME "libsharemind-softfloat-dbg"
    DESCRIPTION "Sharemind SoftFloat library debugging symbols"
    DEB_SECTION "debug"
    DEB_DEPENDS "libsharemind-softfloat (= ${SharemindLibSoftfloat_DEB_lib_PACKAGE_VERSION})"
)
SharemindPackagingFinalize()
