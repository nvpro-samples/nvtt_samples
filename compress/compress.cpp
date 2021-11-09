/*
 * Copyright (c) 2007-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2007-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utilities.h"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <nvtt/nvtt.h>
#include <string.h>

namespace fs = std::filesystem;
using Clock  = std::chrono::high_resolution_clock;

struct MyOutputHandler : public nvtt::OutputHandler
{
  MyOutputHandler(const char* name)
      : total(0)
      , progress(0)
      , percentage(0)
  {
    stream = std::fstream(name, std::ios::out | std::ios::binary);
  }
  virtual ~MyOutputHandler() { stream.close(); }

  void setTotal(uint64_t t)
  {
    // Include the size of the 128-byte DDS header:
    total = t + 128;
  }
  void setDisplayProgress(bool b) { verbose = b; }

  virtual void beginImage(int size, int width, int height, int depth, int face, int miplevel)
  {
    // ignore.
  }

  virtual void endImage()
  {
    // Ignore.
  }

  // Output data.
  virtual bool writeData(const void* data, int size)
  {
    if((!stream.is_open()) || stream.fail())
      return false;

    stream.write(static_cast<const char*>(data), size);

    progress += size;
    int p = int((100 * progress) / total);

    if(verbose && p != percentage)
    {
      assert(p >= 0);

      percentage = p;
      printf("\r%d%%", percentage);
      fflush(stdout);
    }

    return true;
  }

  uint64_t     total;     // Estimated size of the file.
  uint64_t     progress;  // Current number of bytes written.
  int          percentage;
  bool         verbose;
  std::fstream stream;
};

struct MyErrorHandler : public nvtt::ErrorHandler
{
  virtual void error(nvtt::Error e)
  {
    assert(false);
    printf("Error: '%s'\n", nvtt::errorString(e));
  }
};


int main(int argc, char* argv[])
{
  bool alpha            = false;
  bool alpha_set        = false;
  bool alpha_dithering  = false;
  int  alpha_bits       = -1;
  bool normal           = false;
  bool color2normal     = false;
  bool normalizeMipMaps = false;

  bool wrapRepeat = false;
  bool noMipmaps  = false;

  bool fast       = false;
  bool production = false;
  bool highest    = false;

  bool nocuda    = false;
  bool bc1n      = false;
  bool luminance = false;

  nvtt::Format       format       = nvtt::Format_BC1;
  nvtt::MipmapFilter mipmapFilter = nvtt::MipmapFilter_Box;
  bool               rgbm         = false;
  bool               rangescale   = false;

  bool silent    = false;
  bool dds10     = false;
  bool profiling = false;

  fs::path input;
  fs::path output;

  float WeightR = 1.0f;
  float WeightG = 1.0f;
  float WeightB = 1.0f;
  float WeightA = 1.0f;

  // Parse arguments.
  for(int i = 1; i < argc; i++)
  {
    // Input options.
    if(strcmp("-color", argv[i]) == 0)
    {
    }
    else if(strcmp("-alpha", argv[i]) == 0)
    {
      alpha     = true;
      alpha_set = true;
    }
    else if(strcmp("-noalpha", argv[i]) == 0)
    {
      alpha     = false;
      alpha_set = true;
    }
    else if(strcmp("-alpha_dithering", argv[i]) == 0)
    {
      alpha_dithering = true;
      if(i + 1 < argc)
      {
        int bits = atoi(argv[i + 1]);
        if(bits > 0)
        {
          alpha_bits = bits;
          i++;
        }
      }
    }
    else if(strcmp("-normal", argv[i]) == 0)
    {
      normal           = true;
      color2normal     = false;
      normalizeMipMaps = true;
    }
    else if(strcmp("-tonormal", argv[i]) == 0)
    {
      normal           = false;
      color2normal     = true;
      normalizeMipMaps = true;
    }
    else if(strcmp("-clamp", argv[i]) == 0)
    {
    }
    else if(strcmp("-repeat", argv[i]) == 0)
    {
      wrapRepeat = true;
    }
    else if(strcmp("-nomips", argv[i]) == 0)
    {
      noMipmaps = true;
    }
    else if(strcmp("-mipfilter", argv[i]) == 0)
    {
      if(i + 1 == argc)
        break;
      i++;

      if(strcmp("box", argv[i]) == 0)
        mipmapFilter = nvtt::MipmapFilter_Box;
      else if(strcmp("triangle", argv[i]) == 0)
        mipmapFilter = nvtt::MipmapFilter_Triangle;
      else if(strcmp("kaiser", argv[i]) == 0)
        mipmapFilter = nvtt::MipmapFilter_Kaiser;
    }
    else if(strcmp("-rgbm", argv[i]) == 0)
    {
      rgbm = true;
    }
    else if(strcmp("-rangescale", argv[i]) == 0)
    {
      rangescale = true;
    }
    else if(strcmp("-weight_r", argv[i]) == 0)
    {
      i++;
      if(i < argc)
        WeightR = (float)atof(argv[i]);
    }
    else if(strcmp("-weight_g", argv[i]) == 0)
    {
      i++;
      if(i < argc)
        WeightG = (float)atof(argv[i]);
    }
    else if(strcmp("-weight_b", argv[i]) == 0)
    {
      i++;
      if(i < argc)
        WeightB = (float)atof(argv[i]);
    }
    else if(strcmp("-weight_a", argv[i]) == 0)
    {
      i++;
      if(i < argc)
        WeightA = (float)atof(argv[i]);
    }
    // Compression options.
    else if(strcmp("-fast", argv[i]) == 0)
    {
      fast = true;
    }
    else if(strcmp("-production", argv[i]) == 0)
    {
      production = true;
    }
    else if(strcmp("-highest", argv[i]) == 0)
    {
      highest = true;
    }
    else if(strcmp("-nocuda", argv[i]) == 0)
    {
      nocuda = true;
    }
    else if(strcmp("-rgb", argv[i]) == 0)
    {
      format = nvtt::Format_RGB;
    }
    else if(strcmp("-lumi", argv[i]) == 0)
    {
      luminance = true;
      format    = nvtt::Format_RGB;
    }
    else if(strcmp("-bc1", argv[i]) == 0)
    {
      format = nvtt::Format_BC1;
    }
    else if(strcmp("-bc1n", argv[i]) == 0)
    {
      format = nvtt::Format_BC1;
      bc1n   = true;
    }
    else if(strcmp("-bc1a", argv[i]) == 0)
    {
      format = nvtt::Format_BC1a;
    }
    else if(strcmp("-bc2", argv[i]) == 0)
    {
      format = nvtt::Format_BC2;
    }
    else if(strcmp("-bc3", argv[i]) == 0)
    {
      format = nvtt::Format_BC3;
    }
    else if(strcmp("-bc3n", argv[i]) == 0)
    {
      format = nvtt::Format_BC3n;
    }
    else if(strcmp("-bc4", argv[i]) == 0)
    {
      format = nvtt::Format_BC4;
    }
    else if(strcmp("-bc4s", argv[i]) == 0)
    {
      format = nvtt::Format_BC4S;
    }
    else if(strcmp("-ati2", argv[i]) == 0)
    {
      format = nvtt::Format_ATI2;
    }
    else if(strcmp("-bc5", argv[i]) == 0)
    {
      format = nvtt::Format_BC5;
    }
    else if(strcmp("-bc5s", argv[i]) == 0)
    {
      format = nvtt::Format_BC5S;
    }
    else if(strcmp("-bc6", argv[i]) == 0)
    {
      format = nvtt::Format_BC6U;
    }
    else if(strcmp("-bc6s", argv[i]) == 0)
    {
      format = nvtt::Format_BC6S;
    }
    else if(strcmp("-bc7", argv[i]) == 0)
    {
      format = nvtt::Format_BC7;
    }
    else if(strcmp("-bc3_rgbm", argv[i]) == 0)
    {
      format = nvtt::Format_BC3_RGBM;
      rgbm   = true;
    }
    else if(strcmp("-astc_ldr_4x4", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_4x4;
    }
    else if(strcmp("-astc_ldr_5x4", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_5x4;
    }
    else if(strcmp("-astc_ldr_5x5", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_5x5;
    }
    else if(strcmp("-astc_ldr_6x5", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_6x5;
    }
    else if(strcmp("-astc_ldr_6x6", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_6x6;
    }
    else if(strcmp("-astc_ldr_8x5", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_8x5;
    }
    else if(strcmp("-astc_ldr_8x6", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_8x6;
    }
    else if(strcmp("-astc_ldr_8x8", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_8x8;
    }
    else if(strcmp("-astc_ldr_10x5", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_10x5;
    }
    else if(strcmp("-astc_ldr_10x6", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_10x6;
    }
    else if(strcmp("-astc_ldr_10x8", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_10x8;
    }
    else if(strcmp("-astc_ldr_10x10", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_10x10;
    }
    else if(strcmp("-astc_ldr_12x10", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_12x10;
    }
    else if(strcmp("-astc_ldr_12x12", argv[i]) == 0)
    {
      format = nvtt::Format_ASTC_LDR_12x12;
    }
    else if(strcmp("-pause", argv[i]) == 0)
    {
      printf("Press ENTER\n");
      fflush(stdout);
      getchar();
    }

    // Output options
    else if(strcmp("-silent", argv[i]) == 0)
    {
      silent = true;
    }
    else if(strcmp("-dds10", argv[i]) == 0)
    {
      dds10 = true;
    }

    else if(strcmp("-profile", argv[i]) == 0)
    {
      profiling = true;
    }

    else if(argv[i][0] != '-')
    {
      input = argv[i];

      if(i + 1 < argc && argv[i + 1][0] != '-')
      {
        output = argv[i + 1];
      }
      else
      {
        output = input.replace_extension("dds");
      }

      if(output == input)
      {
        output = output.replace_extension("");
        output += "_out.dds";
      }

      break;
    }
    else
    {
      printf("Warning: unrecognized option \"%s\"\n", argv[i]);
    }
  }

  const uint32_t version = nvtt::version();
  const uint32_t major   = version / 100 / 100;
  const uint32_t minor   = (version / 100) % 100;
  const uint32_t rev     = version % 100;


  if(!silent)
  {
    printf("NVIDIA Texture Tools %u.%u.%u - Copyright NVIDIA Corporation 2015 - 2021\n\n", major, minor, rev);
  }

  if(input.empty())
  {
    printf("usage: nvtt_compress [options] infile [outfile.dds]\n\n");

    printf("Input options:\n");
    printf("  -color        The input image is a color map (default).\n");
    printf("  -alpha        The input image has an alpha channel used for transparency.\n");
    printf("  -noalpha      The input image has no alpha channel used for transparency.\n");
    printf(
        "  -alpha_dithering  Enable alpha dithering. Can be followed by a number indicating the number of bits used in "
        "alpha dithering.\n");
    printf("  -normal       The input image is a normal map.\n");
    printf("  -tonormal     Convert input to normal map.\n");
    printf("  -clamp        Clamp wrapping mode (default).\n");
    printf("  -repeat       Repeat wrapping mode.\n");
    printf("  -nomips       Disable mipmap generation.\n");
    printf("  -mipfilter    Mipmap filter. One of the following: box, triangle, kaiser.\n");
    printf("  -rgbm         Transform input to RGBM.\n");
    printf("  -rangescale   Scale image to use entire color range.\n");
    printf("  -weight_r     Weight of R channel, default is 1.\n");
    printf("  -weight_g     Weight of G channel, default is 1.\n");
    printf("  -weight_b     Weight of B channel, default is 1.\n");
    printf(
        "  -weight_a     Weight of A channel, default is 1 when alpha is used, overwritten to 0 when alpha is not "
        "used.\n\n");

    printf("Compression options:\n");
    printf("  -fast         Fast compression.\n");
    printf("  -production   Production compression (higher/slower than default).\n");
    printf("  -highest      Highest-quality compression.\n");
    printf("  -nocuda       Do not use cuda compressor.\n");
    printf("  -rgb          RGBA format\n");
    printf("  -lumi         LUMINANCE format\n");
    printf("  -bc1          BC1 format (DXT1)\n");
    printf("  -bc1n         BC1 normal map format (DXT1nm)\n");
    printf("  -bc1a         BC1 format with binary alpha (DXT1a)\n");
    printf("  -bc2          BC2 format (DXT3)\n");
    printf("  -bc3          BC3 format (DXT5)\n");
    printf("  -bc3n         BC3 normal map format (DXT5nm)\n");
    printf("  -bc4          BC4 format (ATI1)\n");
    printf("  -bc4s         BC4 format (Signed)\n");
    printf("  -ati2         ATI2 format \n");
    printf("  -bc5          BC5 format \n");
    printf("  -bc5s         BC5 format (Signed)\n");
    printf("  -bc6          BC6 format\n");
    printf("  -bc6s         BC6 format (Signed)\n");
    printf("  -bc7          BC7 format\n");
    printf("  -bc3_rgbm     BC3-rgbm format\n");
    printf("  -astc_ldr_4x4 -astc_ldr_5x4 ... -astc_ldr_12x12 ASTC LDR formats\n\n");

    printf("Output options:\n");
    printf("  -silent  \tDo not output progress messages\n");
    printf("  -dds10   \tUse DirectX 10 DDS format (enabled by default for BC6/7 and ASTC)\n");
    printf("  -profile \tShow detailed profiling information\n\n");

    return EXIT_FAILURE;
  }

  bool SNorm = false;


  // Make sure input file exists.
  if(!fs::exists(input))
  {
    fprintf(stderr, "The file '%s' does not exist.\n", input.string().c_str());
    return 1;
  }

  // Set input options.
  nvtt::WrapMode wrapMode = wrapRepeat ? nvtt::WrapMode_Repeat : nvtt::WrapMode_Clamp;

  // no need to resize
#if 1
  nvtt::RoundMode roundMode = nvtt::RoundMode_None;
#else
  nvtt::RoundMode roundMode = (!noMipmaps && format != nvtt::Format_RGB) ? nvtt::RoundMode_ToPreviousPowerOfTwo : nvtt::RoundMode_None;
#endif

  nvtt::CompressionOptions compressionOptions;
  compressionOptions.setFormat(format);

  if(format == nvtt::Format_BC2)
  {
    if(alpha_dithering)
    {
      int bits = 4;
      if(alpha_bits > 0)
        bits = alpha_bits;
      // Dither alpha when using BC2.
      compressionOptions.setPixelFormat(8, 8, 8, bits);
      compressionOptions.setQuantization(/*color dithering*/ false, /*alpha dithering*/ true, /*binary alpha*/ false);
    }
  }
  else if(format == nvtt::Format_BC1a)
  {
    if(alpha_dithering)
    {
      // Binary alpha when using BC1a.
      compressionOptions.setQuantization(/*color dithering*/ false, /*alpha dithering*/ true, /*binary alpha*/ true, 127);
    }
    else
    {
      compressionOptions.setQuantization(/*color dithering*/ false, /*alpha dithering*/ false, /*binary alpha*/ true, 127);
    }
  }
  else if(format == nvtt::Format_RGBA)
  {
    if(luminance)
    {
      compressionOptions.setPixelFormat(8, 0xff, 0, 0, 0);
    }
    else
    {
      // @@ Edit this to choose the desired pixel format:
      // compressionOptions.setPixelType(nvtt::PixelType_Float);
      // compressionOptions.setPixelFormat(16, 16, 16, 16);
      // compressionOptions.setPixelType(nvtt::PixelType_UnsignedNorm);
      // compressionOptions.setPixelFormat(16, 0, 0, 0);

      //compressionOptions.setQuantization(/*color dithering*/true, /*alpha dithering*/false, /*binary alpha*/false);
      //compressionOptions.setPixelType(nvtt::PixelType_UnsignedNorm);
      //compressionOptions.setPixelFormat(5, 6, 5, 0);
      //compressionOptions.setPixelFormat(8, 8, 8, 8);

      // A4R4G4B4
      //compressionOptions.setPixelFormat(16, 0xF00, 0xF0, 0xF, 0xF000);

      //compressionOptions.setPixelFormat(32, 0xFF0000, 0xFF00, 0xFF, 0xFF000000);

      // R10B20G10A2
      //compressionOptions.setPixelFormat(10, 10, 10, 2);

      // DXGI_FORMAT_R11G11B10_FLOAT
      //compressionOptions.setPixelType(nvtt::PixelType_Float);
      //compressionOptions.setPixelFormat(11, 11, 10, 0);
    }
  }
  else if(format == nvtt::Format_BC4S || format == nvtt::Format_BC5S)
  {
    SNorm = true;
    compressionOptions.setPixelType(nvtt::PixelType_SignedNorm);
  }
  else if(format == nvtt::Format_BC6U)
  {
    compressionOptions.setPixelType(nvtt::PixelType_UnsignedFloat);
  }
  else if(format == nvtt::Format_BC6S)
  {
    compressionOptions.setPixelType(nvtt::PixelType_Float);
  }

  if(alpha && alpha_dithering && format != nvtt::Format_BC2 && format != nvtt::Format_BC1a)
  {
    int bits = 8;
    if(alpha_bits > 0)
      bits = alpha_bits;
    compressionOptions.setPixelFormat(8, 8, 8, bits);
    compressionOptions.setQuantization(/*color dithering*/ false, /*alpha dithering*/ true, /*binary alpha*/ false);
  }

  if((fast && production) || (production && highest) || (fast && highest))
  {
    fprintf(stderr, "Please set no more than 1 of fast/production/highest.\n");
    return EXIT_FAILURE;
  }
  if(fast)
  {
    compressionOptions.setQuality(nvtt::Quality_Fastest);
  }
  else if(production)
  {
    compressionOptions.setQuality(nvtt::Quality_Production);
  }
  else if(highest)
  {
    compressionOptions.setQuality(nvtt::Quality_Highest);
  }
  else
  {
    compressionOptions.setQuality(nvtt::Quality_Normal);
  }

  if(bc1n)
  {
    compressionOptions.setColorWeights(WeightR, WeightG, 0);
  }
  else
  {
    compressionOptions.setColorWeights(WeightR, WeightG, WeightB, WeightA);
  }

  MyErrorHandler  errorHandler;
  MyOutputHandler outputHandler(output.string().c_str());
  if(outputHandler.stream.fail())
  {
    fprintf(stderr, "Error opening '%s' for writting\n", output.string().c_str());
    return EXIT_FAILURE;
  }

  // Load input image.
  nvtt::Surface    image;
  nvtt::SurfaceSet images;

  bool mutliInputImage = false;

  nvtt::TextureType textureType(nvtt::TextureType_2D);


  if(!noMipmaps && stringEqualsCaseInsensitive(input.extension().string(), ".dds") && format != nvtt::Format_BC3_RGBM
     && !rgbm && format != nvtt::Format_BC6U && format != nvtt::Format_BC6S)
  {
    if(images.loadDDS(input.string().c_str()))
    {
      textureType = images.GetTextureType();

      image           = images.GetSurface(0, 0, SNorm);
      mutliInputImage = (images.GetMipmapCount() > 1 || images.GetFaceCount() > 1);
    }
  }

  if(image.isNull())
  {
    if(!image.load(input.string().c_str(), 0, SNorm))
    {
      fprintf(stderr, "Error opening input file '%s'.\n", input.string().c_str());
      return EXIT_FAILURE;
    }
    textureType = image.type();
  }

  nvtt::AlphaMode alphaMode = image.alphaMode();
  if(alpha_set)
    alphaMode = alpha ? nvtt::AlphaMode_Transparency : nvtt::AlphaMode_None;

  const auto startTime = Clock::now();

  nvtt::Context context(!nocuda);
  bool          useCuda     = context.isCudaAccelerationEnabled();
  const auto    contextTime = Clock::now();

  if(!silent)
  {
    printf("CUDA acceleration ");
    const float time = timeDiff(startTime, contextTime);
    if(useCuda)
    {
      printf("ENABLED. nvtt::Context() time: %.3f seconds\n\n", time);
    }
    else
    {
      printf("DISABLED. nvtt::Context() time: %.3f seconds\n\n", time);
    }
  }

  context.enableTiming(profiling);
  nvtt::TimingContext* timingContext = context.getTimingContext();
  if(timingContext)
    timingContext->SetDetailLevel(3);

  image.setWrapMode(wrapMode);
  image.setNormalMap(normal);

  if(format == nvtt::Format_BC3_RGBM || rgbm)
  {
    if(rangescale)
    {
      // get color range
      float min_color[3], max_color[3];
      image.range(0, &min_color[0], &max_color[0], -1, 0.0f, timingContext);
      image.range(1, &min_color[1], &max_color[1], -1, 0.0f, timingContext);
      image.range(2, &min_color[2], &max_color[2], -1, 0.0f, timingContext);

      //printf("Color range = %.2f %.2f %.2f\n", max_color[0], max_color[1], max_color[2]);

      float       color_range     = std::max({max_color[0], max_color[1], max_color[2]});
      const float max_color_range = 16.0f;

      if(color_range > max_color_range)
      {
        //printf("Clamping color range %f to %f\n", color_range, max_color_range);
        color_range = max_color_range;
      }

      for(int i = 0; i < 3; i++)
      {
        image.scaleBias(i, 1.0f / color_range, 0.0f, timingContext);
      }
      image.toneMap(nvtt::ToneMapper_Linear, /*parameters=*/NULL, timingContext);  // Clamp without changing the hue.

      // Clamp alpha.
      image.clamp(3, 0.0f, 1.0f, timingContext);
    }

    // To gamma.
    image.toGamma(2.2f, timingContext);

    if(format != nvtt::Format_BC3_RGBM)
    {
      alphaMode = nvtt::AlphaMode_None;
      image.toRGBM(1, 0.15f, timingContext);
    }
  }

  if(format == nvtt::Format_BC6U || format == nvtt::Format_BC6S)
    alphaMode = nvtt::AlphaMode_None;

  image.setAlphaMode(alphaMode);

  int faceCount = mutliInputImage ? images.GetFaceCount() : 1;

  int width  = image.width();
  int height = image.height();
  int depth  = image.depth();

  nvtt::getTargetExtent(&width, &height, &depth, 0, roundMode, textureType, timingContext);

  bool useGPUImageProcess = (size_t)16 * width * height * depth < 100 * 1024 * 1024;

  int mipmapCount = noMipmaps ? 1 : nvtt::countMipmaps(width, height, depth);

  int outputSize = context.estimateSize(image, mipmapCount, compressionOptions) * faceCount;

  outputHandler.setTotal(outputSize);
  outputHandler.setDisplayProgress(!silent);

  nvtt::OutputOptions outputOptions;
  outputOptions.setOutputHandler(&outputHandler);
  outputOptions.setErrorHandler(&errorHandler);

  // Automatically use dds10 if compressing to BC6 or BC7
  if(format == nvtt::Format_BC6U || format == nvtt::Format_BC6S || format == nvtt::Format_BC7
     || (format >= nvtt::Format_ASTC_LDR_4x4 && format <= nvtt::Format_ASTC_LDR_12x12))
  {
    dds10 = true;
  }

  if(dds10)
  {
    outputOptions.setContainer(nvtt::Container_DDS10);
  }

  //// compress procedure

  // If the extents have not changed, then we can use source images for all mipmaps.
  bool canUseSourceImages = (image.width() == width && image.height() == height && image.depth() == depth);

  if(!context.outputHeader(textureType, width, height, depth, mipmapCount, normal || color2normal, compressionOptions, outputOptions))
  {
    fprintf(stderr, "Error writing file header.\n");
    return EXIT_FAILURE;
  }

  // Output images.
  for(int f = 0; f < faceCount; f++)
  {
    int w = width;
    int h = height;
    int d = depth;

    bool useSourceImages = canUseSourceImages;

    if(f > 0)
      images.GetSurface(f, 0, image, SNorm);

    if(useCuda && useGPUImageProcess)
      image.ToGPU(timingContext);

    // To normal map.
    if(color2normal)
    {
      image.toGreyScale(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 0.0f, timingContext);
      image.toNormalMap(1.0f / 1.875f, 0.5f / 1.875f, 0.25f / 1.875f, 0.125f / 1.875f, timingContext);
    }

    // To linear space.
    if(!image.isNormalMap() && !(noMipmaps && canUseSourceImages))
    {
      image.toLinear(2.2f, timingContext);
    }

    // Resize input.
    if(!canUseSourceImages)
      image.resize(w, h, d, nvtt::ResizeFilter_Box, timingContext);

    nvtt::Surface tmp = image;
    if(!image.isNormalMap() && !(noMipmaps && canUseSourceImages))
    {
      tmp.toGamma(2.2f, timingContext);
    }

    context.quantize(tmp, compressionOptions);
    context.compress(tmp, f, 0, compressionOptions, outputOptions);


    for(int m = 1; m < mipmapCount; m++)
    {
      w = std::max(1, w / 2);
      h = std::max(1, h / 2);
      d = std::max(1, d / 2);

      if(useSourceImages)
      {
        if(!mutliInputImage || m >= images.GetMipmapCount())
        {                           // One face is missing in this mipmap level.
          useSourceImages = false;  // If one level is missing, ignore the following source images.
        }
      }

      if(useSourceImages)
      {
        images.GetSurface(f, m, image, SNorm);
        if(useCuda && useGPUImageProcess)
          image.ToGPU(timingContext);
        // For already generated mipmaps, we need to convert to linear.
        if(!image.isNormalMap())
        {
          image.toLinear(2.2f, timingContext);
        }
      }
      else
      {
        if(mipmapFilter == nvtt::MipmapFilter_Kaiser)
        {
          float params[2] = {1.0f /*kaiserStretch*/, 4.0f /*kaiserAlpha*/};
          image.buildNextMipmap(nvtt::MipmapFilter_Kaiser, 3 /*kaiserWidth*/, params, 1, timingContext);
        }
        else
        {
          image.buildNextMipmap(mipmapFilter, 1, timingContext);
        }
      }

      if(image.isNormalMap())
      {
        if(normalizeMipMaps)
        {
          image.normalizeNormalMap(timingContext);
        }
        tmp = image;
      }
      else
      {
        tmp = image;
        tmp.toGamma(2.2f, timingContext);
      }

      context.quantize(tmp, compressionOptions);
      context.compress(tmp, f, m, compressionOptions, outputOptions);
    }
  }


  const auto endTime = Clock::now();

  if(!silent)
  {
    printf("\rTotal processing time: %.3f seconds\n\n", timeDiff(startTime, endTime));
  }

  if(profiling)
  {
    printf("\rDetailed profiling:\n");
    context.getTimingContext()->PrintRecords();
  }

  return EXIT_SUCCESS;
}
