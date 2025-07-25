
```
g++ py_nvbench.cpp                                    \
   -shared -fPIC                                      \
   -I ${HOME}/repos/pybind11/include                  \
   -I ${HOME}/repos/pynvbench/nvbench_dir/include     \
   -I /usr/local/cuda/include                         \
   $(python3-config --includes)                       \
   $(python3-config --libs)                           \
   -L ${HOME}/repos/pynvbench/nvbench_dir/lib/        \
   -lnvbench                                          \
   -Wl,-rpath,${HOME}/repos/pynvbench/nvbench_dir/lib \
   -L /usr/local/cuda/lib64/                          \
   -lcudart                                           \
   -Wl,-rpath,/usr/local/cuda/lib64                   \
   -o _nvbench$(python3-config --extension-suffix)
```
