# Simulating the Snaperz Piston Extender
A simple C++ program for simulating a Snaperz Piston Extender, a specific subclass of piston extenders in Minecraft.

## Building the project
In order to build the project, run the `rebuild.sh` script.
This script contains the following commands, which will build the project in release mode:
```bash
mkdir -p "./build"
cd "./build"
cmake -DCMAKE_BUILD_TYPE=Release ..
make all
```
The project requires that `GCC` supporting C++17, `cmake`, and `make` are installed on your system.

In order to compile in debug mode, replace the third command with:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Linux
You likely already have the required build tools, simply run the commands above.

### Windows
If you are using Windows, the build dependencies can be installed through e.g. `cygwin64`, where the above command is run through the cygwin terminal.

## Running the project
After building the project, the simulation can be run through the following terminal command:
```bash
./build/extender
```
Depending on the size of the extender, this could take a significant amount of time. Be patient!

## Blazingly fast AVX2
Since the dawn of this project the simulation also has AVX2 support, developed by G4me4u. This makes the program slightly less simple but at the same time blazingly fast! This feature requires AVX2 support on your CPU, and will otherwise use the traditional fallback implementation. CPU support is checked by running the command below in the terminal.
```bash
gcc -mavx2 -dM -E - < /dev/null | egrep "SSE|AVX" | sort
```
Your CPU supports AVX2 if the `#define __AVX2__ 1` line is shown in the output.

## Credit

Development into Snaperz extenders spans multiple years. If you played a role but we forgot to list you here, contact us and we will add your name.

- Snaperz
- Space Walker
- MadCloud101
- G4me4u
- Ralp
