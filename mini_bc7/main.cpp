/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
// Miniature sample showing how to load an image file and use CUDA-accelerated
// compression to create a one-surface BC7-compressed DDS file.

#include <iostream>
#include <nvtt/nvtt.h>

int main(int argc, char** argv)
{
  if(argc != 3)
  {
    std::cout << "Miniature sample showing how to convert an image to a one-surface BC7-compressed DDS file using "
                 "nvtt::Surface.\n";
    std::cout << "Usage: nvtt_mini_bc7 infile.png outfile.dds\n";
    return 0;
  }

  // Load the source image into a floating-point RGBA image.
  nvtt::Surface image;
  image.load(argv[1]);

  // Create the compression context; enable CUDA compression, so that
  // CUDA-capable GPUs will use GPU acceleration for compression, with a
  // fallback on other GPUs for CPU compression.
  nvtt::Context context(true);

  // Specify what compression settings to use. In our case the only default
  // to change is that we want to compress to BC7.
  nvtt::CompressionOptions compressionOptions;
  compressionOptions.setFormat(nvtt::Format_BC7);

  // Specify how to output the compressed data. Here, we say to write to a file.
  // We could also use a custom output handler here instead.
  nvtt::OutputOptions outputOptions;
  outputOptions.setFileName(argv[2]);

  // Write the DDS header. Since this uses the BC7 format, this will
  // automatically use the DX10 DDS extension.
  if(!context.outputHeader(image, 1 /* number of mipmaps */, compressionOptions, outputOptions)){
      std::cerr << "Writing the DDS header failed!";
      return 1;
  }

  // Compress and write the compressed data.
  if(!context.compress(image, 0 /* face */, 0 /* mipmap */, compressionOptions, outputOptions)){
      std::cerr << "Compressing and writing the DDS file failed!";
      return 1;
  }

  return 0;
}