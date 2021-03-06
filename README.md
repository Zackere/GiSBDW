# GiSBDW
## Building
1. Open repo root folder following [this](https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=vs-2019) guide.
2. If your favourite CUDA installation is not the only installation on your system, set `OVERRIDE_CUDA_DIR` env variable, e.g. `OVERRIDE_CUDA_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2`. This folder should contain `bin/nvcc`.
3. After cmake is done generating cache (be patient) you should see TreeDepth Project in Solution Explorer. Expand it.
4. Right-click a target you are interested in and set it up as Startup Item.
5. You should now be able to launch and debug as usual by pressing F5.
## Developing
When writing code, few things should be taken into account:
1. Try to install clang-format from [here](https://llvm.org/builds/). Google style is highly desirable when writing in C++.
2. Try to install cpplint from [here](https://github.com/cpplint/cpplint). It will be incorporated later on into the project.
3. C++ sources should be created in src folder and their names should be in [Snake case](https://en.wikipedia.org/wiki/Snake_case) format. C++ headers should have `.hpp` extension and have corresponding `.cpp` file. Cuda source files should have `.cu` extension. Classes composing a module should be contained in a separate subfolder, e.g. `src/graph`.
4. Class/namespace/function names should be in [Pascal case](https://pl.wikipedia.org/wiki/PascalCase).
5. Inside `.cu` sources it is a good idea to start it off like this:<br/>
`// Copyright 2020 GISBDW. All rights reserved.`<br/>
`#include "awesome_header.hpp"`<br/>
`// clang-format on `<br/>
`the rest of the file...`<br/>
This way clang-format won't mess up include order.
6. Named constants in `.cpp` files should be defined as a `constexpr` in a anonymous namespace named in [Pascal case](https://pl.wikipedia.org/wiki/PascalCase) prefixed by `k` e.g. `constexpr  int  kMatrixSize = 4;`
