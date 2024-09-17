/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
// Miniature sample showing how to generate mipmaps for an image using NVTT.
//
// Most common color images on computers (e.g. PNG files) are stored in the
// sRGB color space as of this writing. What this means is that encoded values
// have a nonlinear relationship to the amount of light produced: an sRGB value
// of 255/255 corresponds to about 4.7 times the amount of light as an sRGB
// value of 127/255. A major reason sRGB's used is because it's efficient for
// quantized storage (since our eyes are also nonlinear).
// This means that if we want to resize images so that they look the same
// as resizing textures in real life, we must convert from sRGB to a linear
// color space, resize, then convert back from linear to sRGB.
//
// (You can try this out yourself: in an image editor, create
// a checkerboard of black and white pixels, then move away from the screen so
// that the individual pixels are no longer visible. The gray you see should
// be close to an sRGB gray of 187/255, significantly brighter than 127/255).
//
// To correctly resize an image with transparency, we must also use
// premultiplied alpha (multiply RGB by A). The reason for this is that
// "normal" (unpremultiplied or unassociated alpha) RGBA colors cannot be
// correctly blended without a division somewhere (see e.g. the definition of
// the Porter-Duff over operator for compositing). Premultiplied alpha colors,
// however, can be averaged and the results will be mathematically correct.
//
// Here's an example: suppose we want to perform a 50% blend between two RGBA
// colors, a fully opaque red (1, 0, 0, 1) and an almost invisible color where
// the RGB channels happen to be painted green (0, 1, 0, 0.0001). If we used
// unpremultiplied alpha, the invisible green would leak into the red, and we'd
// get a half-opaque yellow (0.5, 0.5, 0, 0.50005)! Premultiplied alpha gives
// the correct result, a half-opaque red with just a bit of green,
// (1, 0.0001, 0, 0.50005).
//
// This means that before resizing an sRGB image where alpha represents
// transparency, we should convert to linear, then premultiply alpha,
// then resize. To convert back, we unpremultiply colors, then convert back
// to sRGB.
//
// Sometimes alpha doesn't represent transparency, of course (e.g. when it
// stores material information)! In that case, no premultiplication should
// be used.
//
// For more information about premultiplied alpha, see John McDonald's
// blog post Alpha Blending: To Pre or Not To Pre at
// https://developer.nvidia.com/content/alpha-blending-pre-or-not-pre
//
// Mipmapping material textures such as normal maps is of course a whole topic
// in and of itself; for more information on this, please see the tooltips in
// the Texture Tools Exporter.

#include <iostream>
#include <nvtt/nvtt.h>

int main(int argc, char** argv)
{
  if(argc != 3)
  {
    std::cout << "nvtt_mipmap - Miniature sample showing how to generate mipmaps for an image using NVTT.\n";
    std::cout << "Usage: nvtt_mipmap infile.png outfile.dds\n";
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

  // Compute the number of mips.
  const int numMipmaps = image.countMipmaps();

  // Write the DDS header.
  if(!context.outputHeader(image, numMipmaps, compressionOptions, outputOptions))
  {
    std::cerr << "Writing the DDS header failed!";
    return 1;
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

    // Prepare the next mip:

    // Convert to linear premultiplied alpha. Note that toLinearFromSrgb()
    // will clamp HDR images; consider e.g. toLinear(2.2f) instead.
    image.toLinearFromSrgb();
    image.premultiplyAlpha();

    // Resize the image to the next mipmap size.
    // NVTT has several mipmapping filters; Box is the lowest-quality, but
    // also the fastest to use.
    image.buildNextMipmap(nvtt::MipmapFilter_Box);
    // For general image resizing. use image.resize().

    // Convert back to unpremultiplied sRGB.
    image.demultiplyAlpha();
    image.toSrgb();
  }

  return 0;
}