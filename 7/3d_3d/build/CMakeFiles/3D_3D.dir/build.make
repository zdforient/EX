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
CMAKE_SOURCE_DIR = /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/build

# Include any dependencies generated for this target.
include CMakeFiles/3D_3D.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/3D_3D.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/3D_3D.dir/flags.make

CMakeFiles/3D_3D.dir/3D_3D.cpp.o: CMakeFiles/3D_3D.dir/flags.make
CMakeFiles/3D_3D.dir/3D_3D.cpp.o: ../3D_3D.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/3D_3D.dir/3D_3D.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/3D_3D.dir/3D_3D.cpp.o -c /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/3D_3D.cpp

CMakeFiles/3D_3D.dir/3D_3D.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/3D_3D.dir/3D_3D.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/3D_3D.cpp > CMakeFiles/3D_3D.dir/3D_3D.cpp.i

CMakeFiles/3D_3D.dir/3D_3D.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/3D_3D.dir/3D_3D.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/3D_3D.cpp -o CMakeFiles/3D_3D.dir/3D_3D.cpp.s

CMakeFiles/3D_3D.dir/3D_3D.cpp.o.requires:

.PHONY : CMakeFiles/3D_3D.dir/3D_3D.cpp.o.requires

CMakeFiles/3D_3D.dir/3D_3D.cpp.o.provides: CMakeFiles/3D_3D.dir/3D_3D.cpp.o.requires
	$(MAKE) -f CMakeFiles/3D_3D.dir/build.make CMakeFiles/3D_3D.dir/3D_3D.cpp.o.provides.build
.PHONY : CMakeFiles/3D_3D.dir/3D_3D.cpp.o.provides

CMakeFiles/3D_3D.dir/3D_3D.cpp.o.provides.build: CMakeFiles/3D_3D.dir/3D_3D.cpp.o


# Object files for target 3D_3D
3D_3D_OBJECTS = \
"CMakeFiles/3D_3D.dir/3D_3D.cpp.o"

# External object files for target 3D_3D
3D_3D_EXTERNAL_OBJECTS =

3D_3D: CMakeFiles/3D_3D.dir/3D_3D.cpp.o
3D_3D: CMakeFiles/3D_3D.dir/build.make
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
3D_3D: /usr/local/lib/libg2o_core.so
3D_3D: /usr/local/lib/libg2o_stuff.so
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
3D_3D: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
3D_3D: CMakeFiles/3D_3D.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 3D_3D"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/3D_3D.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/3D_3D.dir/build: 3D_3D

.PHONY : CMakeFiles/3D_3D.dir/build

CMakeFiles/3D_3D.dir/requires: CMakeFiles/3D_3D.dir/3D_3D.cpp.o.requires

.PHONY : CMakeFiles/3D_3D.dir/requires

CMakeFiles/3D_3D.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/3D_3D.dir/cmake_clean.cmake
.PHONY : CMakeFiles/3D_3D.dir/clean

CMakeFiles/3D_3D.dir/depend:
	cd /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/7/3d_3d/build/CMakeFiles/3D_3D.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/3D_3D.dir/depend
