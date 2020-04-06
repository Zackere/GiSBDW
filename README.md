# GiSBDW
## Building
1. Open repo root folder following [this](https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=vs-2019) guide.
2. After cmake is done generating cache (be patient) you should see TreeDepth Project in Solution Explorer. Expand it.
3. Right-click a target you are interested in and set it up as Startup Item.
4. You should now be able to launch and debug as usual by pressing F5.
## Developing
When writing code, few things should be taken into account:
1. Try to install clang-format from [here](https://llvm.org/builds/). Google style is highly desirable when writing in C++.
2. Try to install cpplint from [here](https://github.com/cpplint/cpplint). It will be incorporated later on into the project.
3. C++ sources should be created in src folder and their names should be in [Snake case](https://en.wikipedia.org/wiki/Snake_case) format. C++ headers should have `.hpp` extension and have corresponding `.cpp` file. Cuda source files should have `.cu` extension. Classes composing a module should be contained in a separate subfolder, e.g. `src/graph`.
4. Class/namespace/function names should be in [Pascal case]([https://pl.wikipedia.org/wiki/PascalCase](https://pl.wikipedia.org/wiki/PascalCase)).
5. Inside `.cu` sources it is a good idea to start it off like this:
`// Copyright 2020 GISBDW. All rights reserved.`
`#include "awesome_header.hpp"`
`// clang-format on `
`the rest of the file...`
This way clang-format won't mess up include order.
6. Named constants in `.cpp` files should be defined as a `constexpr` in a anonymous namespace named in [Pascal case]([https://pl.wikipedia.org/wiki/PascalCase](https://pl.wikipedia.org/wiki/PascalCase)) prefixed by `k` e.g. `constexpr  int  kMatrixSize = 4;`
