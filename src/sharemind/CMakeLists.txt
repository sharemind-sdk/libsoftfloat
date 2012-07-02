#
# This file is a part of the Sharemind framework.
# Copyright (C) Cybernetica AS
#
# All rights are reserved. Reproduction in whole or part is prohibited
# without the written consent of the copyright owner. The usage of this
# code is subject to the appropriate license agreement.
#

CMAKE_MINIMUM_REQUIRED(VERSION "2.8")
CMAKE_POLICY(SET CMP0015 NEW)
PROJECT("SHAREMIND_LIBSOFTFLOAT" "C")

IF(NOT SHAREMIND_LIB_DIR)
    MESSAGE(FATAL_ERROR "SHAREMIND_LIB_DIR not defined!")
ENDIF(NOT SHAREMIND_LIB_DIR)

IF(NOT SHAREMIND_INCLUDE_DESTINATION)
    MESSAGE(FATAL_ERROR "SHAREMIND_INCLUDE_DESTINATION not defined!")
ENDIF(NOT SHAREMIND_INCLUDE_DESTINATION)

IF(NOT SHAREMIND_GEN_DIR)
    MESSAGE(FATAL_ERROR "SHAREMIND_GEN_DIR not defined!")
ENDIF(NOT SHAREMIND_GEN_DIR)

SET(CMAKE_C_FLAGS "-std=c99 -Wall -Wextra -O2 -fstrict-aliasing -Wstrict-aliasing=1 -fwrapv -Wabi -Wbad-function-cast -Wc++-compat -Wformat -Wswitch-default -Wunused -Wfloat-equal -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wconversion -Wsign-conversion -Wunused-macros")

INCLUDE(CheckCCompilerFlag)
MACRO(FLAGS_MAGIC flags)
    CHECK_C_COMPILER_FLAG("-${flags}" SHAREMIND_HAS_CFLAGS__${flags})
    IF(SHAREMIND_HAS_CFLAGS__${flags})
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -${flags}")
    ENDIF(SHAREMIND_HAS_CFLAGS__${flags})
ENDMACRO(FLAGS_MAGIC)
FLAGS_MAGIC("Weverything")
FLAGS_MAGIC("Wlogical-op")


ADD_DEFINITIONS("-D_XOPEN_SOURCE=500" "-D__STDC_LIMIT_MACROS")

################################################################################
# libsoftfloat
################################################################################

ADD_LIBRARY("libsoftfloat" STATIC "3rdparty/libsoftfloat/softfloat.c")
SET_SOURCE_FILES_PROPERTIES("3rdparty/libsoftfloat/softfloat.c" PROPERTIES
                            COMPILE_FLAGS "-fPIC")
SET_TARGET_PROPERTIES("libsoftfloat" PROPERTIES
                      OUTPUT_NAME "sharemind_softfloat"
                      LIBRARY_OUTPUT_DIRECTORY "${SHAREMIND_LIB_DIR}"
                      VERSION "0.1.0.0"
                      SOVERSION "0.1"
                      COMPILE_FLAGS "-frounding-math -ftrapping-math -fsignaling-nans")
