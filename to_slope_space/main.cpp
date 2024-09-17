/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <iostream>
#include <nvtt/nvtt.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
  if(argc != 3)
  {
    std::cout << "nvtt_to_slope_space - Miniature sample showing how to load a "
                 "normal map, renormalize it, convert to slope-space, and save as BC5.\n";
    std::cout << "Usage: nvtt_to_slope_space infile outfile.dds\n";
    return EXIT_SUCCESS;
  }

  // Load the source normal map into a floating-point RGBA image.
  nvtt::Surface image;
  if(!image.load(argv[1]))
  {
    std::cout << "Loading failed!";
    return EXIT_FAILURE;
  }

  // Convert to splayed normals. We do the processing on the CPU here, but
  // there are a couple of ways to do this using a combination of NVTT's
  // built-in functions, and this can also be done using CUDA with
  // image.gpuData() (which is the same as image.data()).
  // nvtt::Surfaces have four channels, but we only use the first 3.
  // Also note that we don't call image.renomalize() beforehand, since
  // it won't change the result of this step.
  std::array<float*, 3> channels = {image.channel(0), image.channel(1), image.channel(2)};
  const int64_t         height   = static_cast<int64_t>(image.height());
  const size_t          width    = static_cast<size_t>(image.width());
#pragma omp parallel for
  for(int64_t yi = 0; yi < height; yi++)
  {
    for(size_t xi = 0; xi < width; xi++)
    {
      const size_t idx = yi * width + xi;

      const float z = channels[2][idx];
      if(z == 0.0F)
        continue;
      const float slopeSpaceX = (2.0F * channels[0][idx] - 1.0F) / z;
      const float slopeSpaceY = (2.0F * channels[1][idx] - 1.0F) / z;

      // Convert back to coordinates centered at 0.5.
      channels[0][idx] = 0.5F * slopeSpaceX + 0.5F;
      channels[1][idx] = 0.5F * slopeSpaceY + 0.5F;
    }
  }

  // Now compress while generating mipmaps. This will be similar to the NVTT
  // mipmap sample.

  // Create the compression context; enable CUDA compression, so that
  // CUDA-capable GPUs will use GPU acceleration for compression, with a
  // fallback on other systems for CPU compression.
  nvtt::Context context(true);

  // Specify what compression settings to use. In our case the only default
  // to change is that we want to compress to BC5.
  nvtt::CompressionOptions compressionOptions;
  compressionOptions.setFormat(nvtt::Format_BC5);

  // Specify how to output the compressed data. Here, we say to write to a file.
  // We could also use a custom output handler here instead.
  nvtt::OutputOptions outputOptions;
  outputOptions.setFileName(argv[2]);

  // Compute the number of mips.
  const int numMipmaps = image.countMipmaps();

  // Write the DDS header.
  if(!context.outputHeader(image, numMipmaps, compressionOptions, outputOptions))
  {
    std::cerr << "Writing the DDS header failed!";
    return EXIT_FAILURE;
  }

  for(int mip = 0; mip < numMipmaps; mip++)
  {
    // Compress this image and write its data.
    if(!context.compress(image, 0 /* face */, mip, compressionOptions, outputOptions))
    {
      std::cerr << "Compressing and writing the DDS file failed!";
      return 1;
    }

    if(mip == numMipmaps - 1)
      break;

    // Resize the image to the next mipmap size.
    // NVTT has several mipmapping filters - we use Kaiser here.
    // Also note that we resize the slope-space image directly, which is
    // compatible with LEAN mapping.
    image.buildNextMipmap(nvtt::MipmapFilter_Kaiser);
    // For general image resizing, use image.resize().
  }

  return 0;
}