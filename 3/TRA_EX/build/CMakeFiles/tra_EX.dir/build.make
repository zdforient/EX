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
CMAKE_SOURCE_DIR = /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build

# Include any dependencies generated for this target.
include CMakeFiles/tra_EX.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tra_EX.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tra_EX.dir/flags.make

CMakeFiles/tra_EX.dir/tra_EX.cpp.o: CMakeFiles/tra_EX.dir/flags.make
CMakeFiles/tra_EX.dir/tra_EX.cpp.o: ../tra_EX.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tra_EX.dir/tra_EX.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tra_EX.dir/tra_EX.cpp.o -c /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/tra_EX.cpp

CMakeFiles/tra_EX.dir/tra_EX.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tra_EX.dir/tra_EX.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/tra_EX.cpp > CMakeFiles/tra_EX.dir/tra_EX.cpp.i

CMakeFiles/tra_EX.dir/tra_EX.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tra_EX.dir/tra_EX.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/tra_EX.cpp -o CMakeFiles/tra_EX.dir/tra_EX.cpp.s

CMakeFiles/tra_EX.dir/tra_EX.cpp.o.requires:

.PHONY : CMakeFiles/tra_EX.dir/tra_EX.cpp.o.requires

CMakeFiles/tra_EX.dir/tra_EX.cpp.o.provides: CMakeFiles/tra_EX.dir/tra_EX.cpp.o.requires
	$(MAKE) -f CMakeFiles/tra_EX.dir/build.make CMakeFiles/tra_EX.dir/tra_EX.cpp.o.provides.build
.PHONY : CMakeFiles/tra_EX.dir/tra_EX.cpp.o.provides

CMakeFiles/tra_EX.dir/tra_EX.cpp.o.provides.build: CMakeFiles/tra_EX.dir/tra_EX.cpp.o


# Object files for target tra_EX
tra_EX_OBJECTS = \
"CMakeFiles/tra_EX.dir/tra_EX.cpp.o"

# External object files for target tra_EX
tra_EX_EXTERNAL_OBJECTS =

tra_EX: CMakeFiles/tra_EX.dir/tra_EX.cpp.o
tra_EX: CMakeFiles/tra_EX.dir/build.make
tra_EX: /home/zdf/Documents/pkgs/Pangolin/build/src/libpangolin.so
tra_EX: /usr/lib/x86_64-linux-gnu/libOpenGL.so
tra_EX: /usr/lib/x86_64-linux-gnu/libGLX.so
tra_EX: /usr/lib/x86_64-linux-gnu/libGLU.so
tra_EX: /usr/lib/x86_64-linux-gnu/libGLEW.so
tra_EX: /usr/lib/x86_64-linux-gnu/libEGL.so
tra_EX: /usr/lib/x86_64-linux-gnu/libSM.so
tra_EX: /usr/lib/x86_64-linux-gnu/libICE.so
tra_EX: /usr/lib/x86_64-linux-gnu/libX11.so
tra_EX: /usr/lib/x86_64-linux-gnu/libXext.so
tra_EX: /usr/lib/x86_64-linux-gnu/libOpenGL.so
tra_EX: /usr/lib/x86_64-linux-gnu/libGLX.so
tra_EX: /usr/lib/x86_64-linux-gnu/libGLU.so
tra_EX: /usr/lib/x86_64-linux-gnu/libGLEW.so
tra_EX: /usr/lib/x86_64-linux-gnu/libEGL.so
tra_EX: /usr/lib/x86_64-linux-gnu/libSM.so
tra_EX: /usr/lib/x86_64-linux-gnu/libICE.so
tra_EX: /usr/lib/x86_64-linux-gnu/libX11.so
tra_EX: /usr/lib/x86_64-linux-gnu/libXext.so
tra_EX: /usr/lib/x86_64-linux-gnu/libdc1394.so
tra_EX: /usr/lib/x86_64-linux-gnu/libavcodec.so
tra_EX: /usr/lib/x86_64-linux-gnu/libavformat.so
tra_EX: /usr/lib/x86_64-linux-gnu/libavutil.so
tra_EX: /usr/lib/x86_64-linux-gnu/libswscale.so
tra_EX: /usr/lib/x86_64-linux-gnu/libavdevice.so
tra_EX: /usr/lib/libOpenNI.so
tra_EX: /usr/lib/libOpenNI2.so
tra_EX: /usr/lib/x86_64-linux-gnu/libpng.so
tra_EX: /usr/lib/x86_64-linux-gnu/libz.so
tra_EX: /usr/lib/x86_64-linux-gnu/libjpeg.so
tra_EX: /usr/lib/x86_64-linux-gnu/libtiff.so
tra_EX: /usr/lib/x86_64-linux-gnu/libIlmImf.so
tra_EX: /usr/lib/x86_64-linux-gnu/liblz4.so
tra_EX: CMakeFiles/tra_EX.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tra_EX"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tra_EX.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tra_EX.dir/build: tra_EX

.PHONY : CMakeFiles/tra_EX.dir/build

CMakeFiles/tra_EX.dir/requires: CMakeFiles/tra_EX.dir/tra_EX.cpp.o.requires

.PHONY : CMakeFiles/tra_EX.dir/requires

CMakeFiles/tra_EX.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tra_EX.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tra_EX.dir/clean

CMakeFiles/tra_EX.dir/depend:
	cd /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build/CMakeFiles/tra_EX.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tra_EX.dir/depend

