# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/build

# Include any dependencies generated for this target.
include CMakeFiles/ceresex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ceresex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ceresex.dir/flags.make

CMakeFiles/ceresex.dir/ceresex.cpp.o: CMakeFiles/ceresex.dir/flags.make
CMakeFiles/ceresex.dir/ceresex.cpp.o: ../ceresex.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ceresex.dir/ceresex.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ceresex.dir/ceresex.cpp.o -c /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/ceresex.cpp

CMakeFiles/ceresex.dir/ceresex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ceresex.dir/ceresex.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/ceresex.cpp > CMakeFiles/ceresex.dir/ceresex.cpp.i

CMakeFiles/ceresex.dir/ceresex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ceresex.dir/ceresex.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/ceresex.cpp -o CMakeFiles/ceresex.dir/ceresex.cpp.s

CMakeFiles/ceresex.dir/ceresex.cpp.o.requires:

.PHONY : CMakeFiles/ceresex.dir/ceresex.cpp.o.requires

CMakeFiles/ceresex.dir/ceresex.cpp.o.provides: CMakeFiles/ceresex.dir/ceresex.cpp.o.requires
	$(MAKE) -f CMakeFiles/ceresex.dir/build.make CMakeFiles/ceresex.dir/ceresex.cpp.o.provides.build
.PHONY : CMakeFiles/ceresex.dir/ceresex.cpp.o.provides

CMakeFiles/ceresex.dir/ceresex.cpp.o.provides.build: CMakeFiles/ceresex.dir/ceresex.cpp.o


# Object files for target ceresex
ceresex_OBJECTS = \
"CMakeFiles/ceresex.dir/ceresex.cpp.o"

# External object files for target ceresex
ceresex_EXTERNAL_OBJECTS =

ceresex: CMakeFiles/ceresex.dir/ceresex.cpp.o
ceresex: CMakeFiles/ceresex.dir/build.make
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
ceresex: /home/zdf/Documents/pkgs/Pangolin/build/src/libpangolin.so
ceresex: /usr/local/lib/libceres.a
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
ceresex: /usr/lib/x86_64-linux-gnu/libOpenGL.so
ceresex: /usr/lib/x86_64-linux-gnu/libGLX.so
ceresex: /usr/lib/x86_64-linux-gnu/libGLU.so
ceresex: /usr/lib/x86_64-linux-gnu/libGLEW.so
ceresex: /usr/lib/x86_64-linux-gnu/libEGL.so
ceresex: /usr/lib/x86_64-linux-gnu/libSM.so
ceresex: /usr/lib/x86_64-linux-gnu/libICE.so
ceresex: /usr/lib/x86_64-linux-gnu/libX11.so
ceresex: /usr/lib/x86_64-linux-gnu/libXext.so
ceresex: /usr/lib/x86_64-linux-gnu/libOpenGL.so
ceresex: /usr/lib/x86_64-linux-gnu/libGLX.so
ceresex: /usr/lib/x86_64-linux-gnu/libGLU.so
ceresex: /usr/lib/x86_64-linux-gnu/libGLEW.so
ceresex: /usr/lib/x86_64-linux-gnu/libEGL.so
ceresex: /usr/lib/x86_64-linux-gnu/libSM.so
ceresex: /usr/lib/x86_64-linux-gnu/libICE.so
ceresex: /usr/lib/x86_64-linux-gnu/libX11.so
ceresex: /usr/lib/x86_64-linux-gnu/libXext.so
ceresex: /usr/lib/x86_64-linux-gnu/libdc1394.so
ceresex: /usr/lib/x86_64-linux-gnu/libavcodec.so
ceresex: /usr/lib/x86_64-linux-gnu/libavformat.so
ceresex: /usr/lib/x86_64-linux-gnu/libavutil.so
ceresex: /usr/lib/x86_64-linux-gnu/libswscale.so
ceresex: /usr/lib/x86_64-linux-gnu/libavdevice.so
ceresex: /usr/lib/libOpenNI.so
ceresex: /usr/lib/libOpenNI2.so
ceresex: /usr/lib/x86_64-linux-gnu/libpng.so
ceresex: /usr/lib/x86_64-linux-gnu/libz.so
ceresex: /usr/lib/x86_64-linux-gnu/libjpeg.so
ceresex: /usr/lib/x86_64-linux-gnu/libtiff.so
ceresex: /usr/lib/x86_64-linux-gnu/libIlmImf.so
ceresex: /usr/lib/x86_64-linux-gnu/liblz4.so
ceresex: /usr/lib/x86_64-linux-gnu/libglog.so
ceresex: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
ceresex: /usr/lib/x86_64-linux-gnu/libspqr.so
ceresex: /usr/lib/x86_64-linux-gnu/libtbb.so
ceresex: /usr/lib/x86_64-linux-gnu/libcholmod.so
ceresex: /usr/lib/x86_64-linux-gnu/libccolamd.so
ceresex: /usr/lib/x86_64-linux-gnu/libcamd.so
ceresex: /usr/lib/x86_64-linux-gnu/libcolamd.so
ceresex: /usr/lib/x86_64-linux-gnu/libamd.so
ceresex: /usr/lib/x86_64-linux-gnu/liblapack.so
ceresex: /usr/lib/x86_64-linux-gnu/libblas.so
ceresex: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
ceresex: /usr/lib/x86_64-linux-gnu/librt.so
ceresex: /usr/lib/x86_64-linux-gnu/libcxsparse.so
ceresex: /usr/lib/x86_64-linux-gnu/liblapack.so
ceresex: /usr/lib/x86_64-linux-gnu/libblas.so
ceresex: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
ceresex: /usr/lib/x86_64-linux-gnu/librt.so
ceresex: /usr/lib/x86_64-linux-gnu/libcxsparse.so
ceresex: CMakeFiles/ceresex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ceresex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ceresex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ceresex.dir/build: ceresex

.PHONY : CMakeFiles/ceresex.dir/build

CMakeFiles/ceresex.dir/requires: CMakeFiles/ceresex.dir/ceresex.cpp.o.requires

.PHONY : CMakeFiles/ceresex.dir/requires

CMakeFiles/ceresex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ceresex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ceresex.dir/clean

CMakeFiles/ceresex.dir/depend:
	cd /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/6/ceres/build/CMakeFiles/ceresex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ceresex.dir/depend
