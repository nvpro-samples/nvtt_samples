/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.h"
#include <algorithm>
#include <filesystem>
#include <new>
#include <nvtt/nvtt.h>
#include <string>
#include <string.h>
#include <vector>

namespace fs = std::filesystem;
using Clock  = std::chrono::high_resolution_clock;

struct MyErrorHandler : public nvtt::ErrorHandler
{
  virtual void error(nvtt::Error e)
  {
#if _DEBUG
    nvDebugBreak();
#endif
    printf("Error: '%s'\n", nvtt::errorString(e));
  }
};

struct FileNamePair
{
  fs::path       input;
  fs::path       output;
  std::uintmax_t inputSize;
};

static void GenFileList(const std::string& inDir, const std::string& outDir, std::vector<FileNamePair>& fileList)
{
  for(auto const& dir_entry : fs::directory_iterator(inDir))
  {
    if(!dir_entry.is_regular_file())
      continue;

    fs::path filename = dir_entry.path().filename();

    FileNamePair p1;
    p1.input     = fs::path(inDir) / filename;
    p1.output    = fs::path(outDir) / filename.replace_extension("dds");
    p1.inputSize = dir_entry.file_size();

    fileList.push_back(p1);
  }
}

static void ClearLists(nvtt::BatchList& batchList, std::vector<nvtt::Surface*>& SurfaceList, std::vector<nvtt::OutputOptions*>& OutputOptionsList)
{
  for(unsigned i = 0; i < SurfaceList.size(); i++)
  {
    delete SurfaceList[i];
  }
  SurfaceList.clear();

  for(unsigned i = 0; i < OutputOptionsList.size(); i++)
  {
    delete OutputOptionsList[i];
  }
  OutputOptionsList.clear();

  batchList.Clear();
}

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

  bool silent = false;
  bool dds10  = false;

  float WeightR = 1.0f;
  float WeightG = 1.0f;
  float WeightB = 1.0f;
  float WeightA = 1.0f;

  const char* inStr  = 0;
  const char* outStr = 0;

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

    else if(argv[i][0] != '-')
    {
      inStr = argv[i];

      if(i + 1 < argc && argv[i + 1][0] != '-')
      {
        outStr = argv[i + 1];
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
    printf("NVIDIA Texture Tools %u.%u.%u - Copyright NVIDIA Corporation 2015 - 2021\n", major, minor, rev);
  }

  if(inStr == 0)
  {
    printf("usage: nvtt_batchcompress [options] infile(or dir) [outfile(or dir)]\n\n");

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
    printf("  -dds10   \tUse DirectX 10 DDS format (enabled by default for BC6/7 and ASTC)\n\n");

    return EXIT_FAILURE;
  }

  if(!fs::exists(fs::path(inStr)))
  {
    printf("The input directory %s did not exist.\n", inStr);
    return 0;
  }

  bool isDir = fs::is_directory(fs::path(inStr));

  std::vector<FileNamePair> FileList;
  if(isDir)
  {
    std::string inDir = inStr;
    std::string outDir;

    if(outStr)
      outDir = outStr;
    else
      outDir = inDir + "_out";

    if(!fs::exists(outDir))
    {
      fs::create_directory(outDir);
    }
    GenFileList(inDir, outDir, FileList);
  }
  else
  {
    FileNamePair p1;
    p1.input = inStr;

    if(outStr)
      p1.output = outStr;
    else
    {
      p1.output = p1.input.replace_extension("dds");
    }

    if(p1.output == p1.input)
    {
      p1.output = p1.output.replace_extension("");
      p1.output += "_out.dds";
    }

    FileList.push_back(p1);
  }

  bool SNorm = false;

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

  // Automatically use dds10 if compressing to BC6 or BC7
  if(format == nvtt::Format_BC6U || format == nvtt::Format_BC6S || format == nvtt::Format_BC7
     || (format >= nvtt::Format_ASTC_LDR_4x4 && format <= nvtt::Format_ASTC_LDR_12x12))
  {
    dds10 = true;
  }

  MyErrorHandler errorHandler;

  nvtt::Context context(!nocuda);
  bool          useCuda = context.isCudaAccelerationEnabled();

  if(!silent)
  {
    printf("CUDA acceleration ");
    if(useCuda)
    {
      printf("ENABLED\n\n");
    }
    else
    {
      printf("DISABLED\n\n");
    }
  }

  /// ToDo
  unsigned long long batchSizeLimit = 104857600;

  unsigned long long                curBatchSize = 0;
  nvtt::BatchList                   batchList;
  std::vector<nvtt::Surface*>       SurfaceList;
  std::vector<nvtt::OutputOptions*> OutputOptionsList;
  std::vector<std::string>          compressingList;

  const auto startTime = Clock::now();

  unsigned i = 0;
  while(i < FileList.size())
  {
    for(; i < FileList.size(); i++)
    {
      if(curBatchSize + FileList[i].inputSize > batchSizeLimit && curBatchSize > 0)
        break;

      const fs::path&   input    = FileList[i].input;
      const std::string inputStr = input.string();

      nvtt::Surface    image;
      nvtt::SurfaceSet images;

      bool mutliInputImage = false;

      nvtt::TextureType textureType;

      if(!noMipmaps && stringEqualsCaseInsensitive(input.extension().string(), ".dds")
         && format != nvtt::Format_BC3_RGBM && !rgbm && format != nvtt::Format_BC6U && format != nvtt::Format_BC6S)
      {
        if(images.loadDDS(inputStr.c_str()))
        {
          textureType = images.GetTextureType();

          image           = images.GetSurface(0, 0, SNorm);
          mutliInputImage = (images.GetMipmapCount() > 1 || images.GetFaceCount() > 1);
        }
      }

      if(image.isNull())
      {
        if(!image.load(inputStr.c_str(), 0, SNorm))
        {
          fprintf(stderr, "Error opening input file '%s'.\n", inputStr.c_str());
          return EXIT_FAILURE;
        }
        textureType = image.type();
      }

      image.setWrapMode(wrapMode);
      image.setNormalMap(normal);

      nvtt::AlphaMode alphaMode = image.alphaMode();
      if(alpha_set)
        alphaMode = alpha ? nvtt::AlphaMode_Transparency : nvtt::AlphaMode_None;

      if(format == nvtt::Format_BC3_RGBM || rgbm)
      {
        if(rangescale)
        {
          // get color range
          float min_color[3], max_color[3];
          image.range(0, &min_color[0], &max_color[0], -1, 0.0f);
          image.range(1, &min_color[1], &max_color[1], -1, 0.0f);
          image.range(2, &min_color[2], &max_color[2], -1, 0.0f);

          //printf("Color range = %.2f %.2f %.2f\n", max_color[0], max_color[1], max_color[2]);

          float       color_range     = std::max({max_color[0], max_color[1], max_color[2]});
          const float max_color_range = 16.0f;

          if(color_range > max_color_range)
          {
            //printf("Clamping color range %f to %f\n", color_range, max_color_range);
            color_range = max_color_range;
          }
          //color_range = max_color_range;  // Use a fixed color range for now.

          for(int i = 0; i < 3; i++)
          {
            image.scaleBias(i, 1.0f / color_range, 0.0f);
          }
          image.toneMap(nvtt::ToneMapper_Linear, /*parameters=*/NULL);  // Clamp without changing the hue.

          // Clamp alpha.
          image.clamp(3, 0.0f, 1.0f);
        }

        // To gamma.
        image.toGamma(2.2f);

        if(format != nvtt::Format_BC3_RGBM)
        {
          alphaMode = nvtt::AlphaMode_None;
          image.toRGBM(1, 0.15f);
        }
      }

      if(format == nvtt::Format_BC6S || format == nvtt::Format_BC6U)
        alphaMode = nvtt::AlphaMode_None;

      image.setAlphaMode(alphaMode);

      int faceCount = mutliInputImage ? images.GetFaceCount() : 1;


      int width  = image.width();
      int height = image.height();
      int depth  = image.depth();

      nvtt::getTargetExtent(&width, &height, &depth, 0, roundMode, textureType);

      int mipmapCount = noMipmaps ? 1 : nvtt::countMipmaps(width, height, depth);

      nvtt::OutputOptions* outputOptions = new nvtt::OutputOptions;
      outputOptions->setErrorHandler(&errorHandler);
      outputOptions->setFileName(FileList[i].output.string().c_str());

      compressingList.push_back(FileList[i].input.string().c_str());

      if(dds10)
      {
        outputOptions->setContainer(nvtt::Container_DDS10);
      }

      //// compress procedure

      // If the extents have not changed, then we can use source images for all mipmaps.
      bool canUseSourceImages = (image.width() == width && image.height() == height && image.depth() == depth);

      if(!context.outputHeader(textureType, width, height, depth, mipmapCount, normal || color2normal, compressionOptions, *outputOptions))
      {
        fprintf(stderr, "Error writing file header %s.\n", FileList[i].output.string().c_str());
        delete outputOptions;
        continue;
      }

      OutputOptionsList.push_back(outputOptions);

      // Output images.
      for(int f = 0; f < faceCount; f++)
      {
        int w = width;
        int h = height;
        int d = depth;

        bool useSourceImages = canUseSourceImages;

        if(f > 0)
          images.GetSurface(f, 0, image, SNorm);

        if(useCuda)
          image.ToGPU();

        // To normal map.
        if(color2normal)
        {
          image.toGreyScale(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 0.0f);
          image.toNormalMap(1.0f / 1.875f, 0.5f / 1.875f, 0.25f / 1.875f, 0.125f / 1.875f);
        }

        // To linear space.
        if(!image.isNormalMap() && !(noMipmaps && canUseSourceImages))
        {
          image.toLinear(2.2f);
        }

        // Resize input.
        if(!canUseSourceImages)
          image.resize(w, h, d, nvtt::ResizeFilter_Box);

        nvtt::Surface tmp = image;
        if(!image.isNormalMap() && !(noMipmaps && canUseSourceImages))
        {
          tmp.toGamma(2.2f);
        }

        context.quantize(tmp, compressionOptions);
        nvtt::Surface* surf = new nvtt::Surface(tmp);
        SurfaceList.push_back(surf);
        batchList.Append(surf, f, 0, outputOptions);

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
            if(useCuda)
              image.ToGPU();
            // For already generated mipmaps, we need to convert to linear.
            if(!image.isNormalMap())
            {
              image.toLinear(2.2f);
            }
          }
          else
          {
            if(mipmapFilter == nvtt::MipmapFilter_Kaiser)
            {
              float params[2] = {1.0f /*kaiserStretch*/, 4.0f /*kaiserAlpha*/};
              image.buildNextMipmap(nvtt::MipmapFilter_Kaiser, 3 /*kaiserWidth*/, params, 1);
            }
            else
            {
              image.buildNextMipmap(mipmapFilter, 1);
            }
          }

          if(image.isNormalMap())
          {
            if(normalizeMipMaps)
            {
              image.normalizeNormalMap();
            }
            tmp = image;
          }
          else
          {
            tmp = image;
            tmp.toGamma(2.2f);
          }

          context.quantize(tmp, compressionOptions);
          nvtt::Surface* surf = new nvtt::Surface(tmp);
          SurfaceList.push_back(surf);
          batchList.Append(surf, f, m, outputOptions);
        }
      }

      curBatchSize += FileList[i].inputSize;
    }

    if(compressingList.size() == 0)
      continue;

    printf("Compressing the following files:\n");
    for(unsigned i = 0; i < compressingList.size(); i++)
    {
      printf("%s\n", compressingList[i].data());
    }
    printf("\n");

    context.compress(batchList, compressionOptions);

    curBatchSize = 0;
    ClearLists(batchList, SurfaceList, OutputOptionsList);
    compressingList.clear();
  }

  const auto endTime = Clock::now();

  if(!silent)
  {
    printf("\rtime taken: %.3f seconds\n\n", timeDiff(startTime, endTime));
  }
}
