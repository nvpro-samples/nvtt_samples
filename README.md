# NVTT 3 Samples

This repository contains a number of samples showing how to use [NVTT 3](https://developer.nvidia.com/gpu-accelerated-texture-compression), a GPU-accelerated texture compression and image processing library.

This includes several small samples intended as tutorials:

* [**mini_bc7**](mini_bc7) shows how to load an image and turn it into a one-mipmap DDS file using BC7 block compression. It's small enough that it can be rewritten in 250 characters (without error checking):

```c++
#include<nvtt/nvtt.h>
int main(int n,char**v){if(n==3){nvtt::Surface i;i.load(v[1]);nvtt::Context c(1);nvtt::CompressionOptions o;o.setFormat(nvtt::Format_BC7);nvtt::OutputOptions p;p.setFileName(v[2]);c.outputHeader(i,1,o,p);c.compress(i,0,0,o,p);}}
```

* [**mipmap**](mipmap) shows how to generate mipmaps from an image, and describes linear-space filtering and premultiplied alpha.
* [**cuda_input**](cuda_input) shows how to use NVTT 3's low-level `GPUInputBuffer` API to compress a texture directly from a GPU buffer. This API accelerates compression by avoiding CPU-GPU data transfers.
* [**c_wrapper_demo**](c_wrapper_demo) shows how to use NVTT 3's C wrapper, which allows other compilers and programming languages to use NVTT 3. It covers the high-level and low-level APIs.

This also includes the source code for several tools from NVTT 3 ported to use the nvpro-samples framework, which show how to use almost all of the functionality in NVTT 3:

* [**compress**](compress) is a general command-line tool for compressing images to DDS files, and uses similar flags as the [Texture Tools Exporter's](https://developer.nvidia.com/nvidia-texture-tools-exporter) command-line interface. It supports many types of images, including normal maps.
* [**batchCompress**](batchCompress) can be used to compress multiple files at once.
* [**decompress**](decompress) decompresses DDS files to other formats.
* [**imgdiff**](imgdiff) reports error metrics between image files.

For comprehensive API documentation, please see the `docs/` folder in the NVTT 3 distribution.

## Build Instructions

To build these samples, you'll need the [NVTT 3 SDK](https://developer.nvidia.com/gpu-accelerated-texture-compression). For `cuda_input`, you'll also need the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Then do one of the following:

- To clone all NVIDIA DesignWorks Samples, clone https://github.com/nvpro-samples/build_all, then run one of the `clone_all` scripts in that directory.
- Or to get the files for this sample without the other samples, clone this repository as well as https://github.com/nvpro-samples/nvpro_core into a single directory.

You can then use [CMake](https://cmake.org/) to generate and subsequently build the project.