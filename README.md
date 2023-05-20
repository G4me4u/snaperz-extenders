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
The project requires that `GCC` supporting C++14, `cmake`, and `make` are installed on your system.

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

## Credit

TODO
