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
include CMakeFiles/ex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ex.dir/flags.make

CMakeFiles/ex.dir/ex.cpp.o: CMakeFiles/ex.dir/flags.make
CMakeFiles/ex.dir/ex.cpp.o: ../ex.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ex.dir/ex.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ex.dir/ex.cpp.o -c /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/ex.cpp

CMakeFiles/ex.dir/ex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ex.dir/ex.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/ex.cpp > CMakeFiles/ex.dir/ex.cpp.i

CMakeFiles/ex.dir/ex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ex.dir/ex.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/ex.cpp -o CMakeFiles/ex.dir/ex.cpp.s

CMakeFiles/ex.dir/ex.cpp.o.requires:

.PHONY : CMakeFiles/ex.dir/ex.cpp.o.requires

CMakeFiles/ex.dir/ex.cpp.o.provides: CMakeFiles/ex.dir/ex.cpp.o.requires
	$(MAKE) -f CMakeFiles/ex.dir/build.make CMakeFiles/ex.dir/ex.cpp.o.provides.build
.PHONY : CMakeFiles/ex.dir/ex.cpp.o.provides

CMakeFiles/ex.dir/ex.cpp.o.provides.build: CMakeFiles/ex.dir/ex.cpp.o


# Object files for target ex
ex_OBJECTS = \
"CMakeFiles/ex.dir/ex.cpp.o"

# External object files for target ex
ex_EXTERNAL_OBJECTS =

ex: CMakeFiles/ex.dir/ex.cpp.o
ex: CMakeFiles/ex.dir/build.make
ex: /home/zdf/Documents/pkgs/Pangolin/build/src/libpangolin.so
ex: /usr/lib/x86_64-linux-gnu/libOpenGL.so
ex: /usr/lib/x86_64-linux-gnu/libGLX.so
ex: /usr/lib/x86_64-linux-gnu/libGLU.so
ex: /usr/lib/x86_64-linux-gnu/libGLEW.so
ex: /usr/lib/x86_64-linux-gnu/libEGL.so
ex: /usr/lib/x86_64-linux-gnu/libSM.so
ex: /usr/lib/x86_64-linux-gnu/libICE.so
ex: /usr/lib/x86_64-linux-gnu/libX11.so
ex: /usr/lib/x86_64-linux-gnu/libXext.so
ex: /usr/lib/x86_64-linux-gnu/libOpenGL.so
ex: /usr/lib/x86_64-linux-gnu/libGLX.so
ex: /usr/lib/x86_64-linux-gnu/libGLU.so
ex: /usr/lib/x86_64-linux-gnu/libGLEW.so
ex: /usr/lib/x86_64-linux-gnu/libEGL.so
ex: /usr/lib/x86_64-linux-gnu/libSM.so
ex: /usr/lib/x86_64-linux-gnu/libICE.so
ex: /usr/lib/x86_64-linux-gnu/libX11.so
ex: /usr/lib/x86_64-linux-gnu/libXext.so
ex: /usr/lib/x86_64-linux-gnu/libdc1394.so
ex: /usr/lib/x86_64-linux-gnu/libavcodec.so
ex: /usr/lib/x86_64-linux-gnu/libavformat.so
ex: /usr/lib/x86_64-linux-gnu/libavutil.so
ex: /usr/lib/x86_64-linux-gnu/libswscale.so
ex: /usr/lib/x86_64-linux-gnu/libavdevice.so
ex: /usr/lib/libOpenNI.so
ex: /usr/lib/libOpenNI2.so
ex: /usr/lib/x86_64-linux-gnu/libpng.so
ex: /usr/lib/x86_64-linux-gnu/libz.so
ex: /usr/lib/x86_64-linux-gnu/libjpeg.so
ex: /usr/lib/x86_64-linux-gnu/libtiff.so
ex: /usr/lib/x86_64-linux-gnu/libIlmImf.so
ex: /usr/lib/x86_64-linux-gnu/liblz4.so
ex: CMakeFiles/ex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ex.dir/build: ex

.PHONY : CMakeFiles/ex.dir/build

CMakeFiles/ex.dir/requires: CMakeFiles/ex.dir/ex.cpp.o.requires

.PHONY : CMakeFiles/ex.dir/requires

CMakeFiles/ex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ex.dir/clean

CMakeFiles/ex.dir/depend:
	cd /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build /home/zdf/Documents/GITHUB/EX_SLAM_Practice/3/TRA_EX/build/CMakeFiles/ex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ex.dir/depend

