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
CMAKE_SOURCE_DIR = /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/build

# Include any dependencies generated for this target.
include CMakeFiles/optical.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/optical.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/optical.dir/flags.make

CMakeFiles/optical.dir/optical_flow.cpp.o: CMakeFiles/optical.dir/flags.make
CMakeFiles/optical.dir/optical_flow.cpp.o: ../optical_flow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/optical.dir/optical_flow.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/optical.dir/optical_flow.cpp.o -c /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/optical_flow.cpp

CMakeFiles/optical.dir/optical_flow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/optical.dir/optical_flow.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/optical_flow.cpp > CMakeFiles/optical.dir/optical_flow.cpp.i

CMakeFiles/optical.dir/optical_flow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/optical.dir/optical_flow.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/optical_flow.cpp -o CMakeFiles/optical.dir/optical_flow.cpp.s

CMakeFiles/optical.dir/optical_flow.cpp.o.requires:

.PHONY : CMakeFiles/optical.dir/optical_flow.cpp.o.requires

CMakeFiles/optical.dir/optical_flow.cpp.o.provides: CMakeFiles/optical.dir/optical_flow.cpp.o.requires
	$(MAKE) -f CMakeFiles/optical.dir/build.make CMakeFiles/optical.dir/optical_flow.cpp.o.provides.build
.PHONY : CMakeFiles/optical.dir/optical_flow.cpp.o.provides

CMakeFiles/optical.dir/optical_flow.cpp.o.provides.build: CMakeFiles/optical.dir/optical_flow.cpp.o


# Object files for target optical
optical_OBJECTS = \
"CMakeFiles/optical.dir/optical_flow.cpp.o"

# External object files for target optical
optical_EXTERNAL_OBJECTS =

optical: CMakeFiles/optical.dir/optical_flow.cpp.o
optical: CMakeFiles/optical.dir/build.make
optical: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
optical: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
optical: CMakeFiles/optical.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable optical"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/optical.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/optical.dir/build: optical

.PHONY : CMakeFiles/optical.dir/build

CMakeFiles/optical.dir/requires: CMakeFiles/optical.dir/optical_flow.cpp.o.requires

.PHONY : CMakeFiles/optical.dir/requires

CMakeFiles/optical.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/optical.dir/cmake_clean.cmake
.PHONY : CMakeFiles/optical.dir/clean

CMakeFiles/optical.dir/depend:
	cd /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8 /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8 /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/8/build/CMakeFiles/optical.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/optical.dir/depend

