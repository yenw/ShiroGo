## ShiroGo V2

## Usage

### 1 Compile `MXNet C++` Library

#### 1.1 Requirements
* GCC or Clang
* Ninja
* CMake
* optional: CUDA
* The program has been tested on Linux and macOS.

#### 1.2 Download MXNet
```bash
# download mxnet
mkdir -p ~/build/lib/
cd ~/build/lib/
wget https://github.com/apache/incubator-mxnet/releases/download/1.6.0/apache-mxnet-src-1.6.0-incubating.tar.gz
tar -xzvf apache-mxnet-src-1.6.0-incubating.tar.gz
mv apache-mxnet-src-1.6.0-incubating mxnet
cd mxnet
```

#### 1.3.1 MacOS CPU
```bash
# cmake
cp CMakeLists_MacOS.txt CMakeLists.txt
mkdir build && cd build
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_OPENMP=0 \
      -DUSE_CUDA=0 \
      -DUSE_OPENCV=0 \
      -DUSE_JEMALLOC=0 \
      -DUSE_CPP_PACKAGE=1 \
      -DUSE_BLAS=apple \
      -DUSE_F16C=0 \
      -GNinja
```

#### 1.3.2 Linux CPU
```bash
# cmake
cp CMakeLists_Linux.txt CMakeLists.txt
mkdir build && cd build
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_OPENMP=1 \
      -DUSE_CUDA=0 \
      -DUSE_OPENCV=0 \
      -DUSE_JEMALLOC=0 \
      -DUSE_CPP_PACKAGE=1 \
      -DUSE_F16C=0 \
      -GNinja
```

#### 1.3.3 Linux GPU
```bash
# cmake
cp CMakeLists_Linux.txt CMakeLists.txt
mkdir build && cd build
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_OPENMP=1 \
      -DUSE_CUDA=1 \
      -DUSE_CUDNN=1 \
      -DUSE_OPENCV=0 \
      -DUSE_JEMALLOC=0 \
      -DUSE_CPP_PACKAGE=1 \
      -DUSE_F16C=0 \
      -GNinja
```

### 1.4 Make
```bash
rm -rf lib && mkdir lib
cd build && ninja -v 
cp ./libmxnet* ../lib
```

### 1.5 Export mxnet 

#### 1.5.1 MacOS
```bash
export DYLD_LIBRARY_PATH=/Users/`yourname`/build/lib/mxnet/lib
```

#### 1.5.2 Linux
```bash
export LD_LIBRARY_PATH=/home/`yourname`/build/lib/mxnet/lib
```

### 2 Compile `ShiroGo`

```bash
git clone https://github.com/yenw/ShiroGo.git
cd ShiroGo
mkdir -p build && cd build && cmake .. && make
```

### 3 Set Selfplay Task

```bash
cp -r tool/cr* ./build
cd build
# edit cr_auto
# ...
# ...
```

### 4 Run Selfplay

```bash
./cr_nohup_out
```

### 5 Watch Selfplay

```bash
./cr_watch
```

### NN Example
* [NN Example block=2, filter=64.pdf](https://github.com/yenw/ShiroGo/blob/master/NN%20Example%20block%3D2%2C%20filter%3D64.pdf)

