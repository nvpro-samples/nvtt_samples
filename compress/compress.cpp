/*
 * Copyright (c) 2007-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2007-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.h"
#include <algorithm>
#include <filesystem>
#include <limits>
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
  std::uintmax_t inputSize = 0;
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
    p1.output    = fs::path(outDir) / fs::path(filename).replace_extension("dds");
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

// Tries to read a nonnegative integer from argument i in the argument list.
// If this fails, prints an error message, leaves the value unchanged,
// and returns false.
static bool tryParseInt(int& value, int i, int argc, char* argv[], const char* argumentNameForMessages)
{
  if(i >= argc)
  {
    printf("%s was at the end of the argument list; it must be followed by a nonnegative integer.", argumentNameForMessages);
    return false;
  }
  char*      end_ptr;
  const long read_value = strtol(argv[i], &end_ptr, 10);
  if(read_value < 0 || read_value >= std::numeric_limits<int>::max() || end_ptr == argv[i])
  {
    printf("%s was followed by a negative, out-of-range, or unparseable integer (%s).", argumentNameForMessages, argv[i]);
    return false;
  }
  value = read_value;
  return true;
}

// Returns true iff argValue is equal to -flagName or --flagName.
// This makes it so that Texture Tools Exporter command lines
// (which use --) often work with nvtt_compress, while being compatible with
// previous nvtt_compress versions (which use
// Both strings must be null-terminated.
static bool argMatches(const char* flagName, const char* argValue)
{
  size_t firstCharAfterDashes = 0;
  while(argValue[firstCharAfterDashes] == '-' && firstCharAfterDashes < 2)
  {
    firstCharAfterDashes++;
  }
  // Note that firstCharAfterDashes might now point to the null terminator.
  // This is OK; it means that flagName "" will match "-" and "--".
  return strcmp(flagName, argValue + firstCharAfterDashes) == 0;
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

  bool wrapRepeat      = false;
  int  maxMipCount     = std::numeric_limits<int>::max();
  int  minMipSize      = 1;
  bool mipGammaCorrect = true;

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

  const char* inStr  = nullptr;
  const char* outStr = nullptr;

  float WeightR = 1.0f;
  float WeightG = 1.0f;
  float WeightB = 1.0f;
  float WeightA = 1.0f;

  // Parse arguments.
  for(int i = 1; i < argc; i++)
  {
    // Input options.
    if(argMatches("color", argv[i]))
    {
    }
    else if(argMatches("alpha", argv[i]))
    {
      alpha     = true;
      alpha_set = true;
    }
    else if(argMatches("noalpha", argv[i]))
    {
      alpha     = false;
      alpha_set = true;
    }
    else if(argMatches("alpha_dithering", argv[i]))
    {
      alpha_dithering = true;
      // Since the number after alpha_dithering is optional,
      // this doesn't use tryParseInt():
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
    else if(argMatches("normal", argv[i]))
    {
      normal           = true;
      color2normal     = false;
      normalizeMipMaps = true;
    }
    else if(argMatches("tonormal", argv[i]))
    {
      normal           = false;
      color2normal     = true;
      normalizeMipMaps = true;
    }
    else if(argMatches("clamp", argv[i]))
    {
    }
    else if(argMatches("repeat", argv[i]))
    {
      wrapRepeat = true;
    }
    else if(argMatches("nomips", argv[i]) || argMatches("no-mips", argv[i]))
    {
      maxMipCount = 1;
    }
    else if(argMatches("max-mip-count", argv[i]))
    {
      i++;
      if(!tryParseInt(maxMipCount, i, argc, argv, "max-mip-count"))
        return EXIT_FAILURE;
    }
    else if(argMatches("min-mip-size", argv[i]))
    {
      i++;
      if(!tryParseInt(minMipSize, i, argc, argv, "min-mip-size"))
        return EXIT_FAILURE;
    }
    else if(argMatches("mipfilter", argv[i]) || argMatches("mip-filter", argv[i]))
    {
      if(i + 1 >= argc)
        break;
      i++;

      if(strcmp("box", argv[i]) == 0)
        mipmapFilter = nvtt::MipmapFilter_Box;
      else if(strcmp("triangle", argv[i]) == 0)
        mipmapFilter = nvtt::MipmapFilter_Triangle;
      else if(strcmp("kaiser", argv[i]) == 0)
        mipmapFilter = nvtt::MipmapFilter_Kaiser;
    }
    else if(argMatches("no-mip-gamma-correct", argv[i]))
    {
      mipGammaCorrect = false;
    }
    else if(argMatches("rgbm", argv[i]))
    {
      rgbm = true;
    }
    else if(argMatches("rangescale", argv[i]))
    {
      rangescale = true;
    }
    else if(argMatches("weight_r", argv[i]) || argMatches("weight-r", argv[i]))
    {
      i++;
      if(i < argc)
        WeightR = (float)atof(argv[i]);
    }
    else if(argMatches("weight_g", argv[i]) || argMatches("weight-g", argv[i]))
    {
      i++;
      if(i < argc)
        WeightG = (float)atof(argv[i]);
    }
    else if(argMatches("weight_b", argv[i]) || argMatches("weight-b", argv[i]))
    {
      i++;
      if(i < argc)
        WeightB = (float)atof(argv[i]);
    }
    else if(argMatches("weight_a", argv[i]) || argMatches("weight-a", argv[i]))
    {
      i++;
      if(i < argc)
        WeightA = (float)atof(argv[i]);
    }
    // Compression options.
    else if(argMatches("fast", argv[i]))
    {
      fast = true;
    }
    else if(argMatches("production", argv[i]))
    {
      production = true;
    }
    else if(argMatches("highest", argv[i]))
    {
      highest = true;
    }
    else if(argMatches("nocuda", argv[i]) || argMatches("no-cuda", argv[i]))
    {
      nocuda = true;
    }
    else if(argMatches("rgb", argv[i]))
    {
      format = nvtt::Format_RGB;
    }
    else if(argMatches("lumi", argv[i]))
    {
      luminance = true;
      format    = nvtt::Format_RGB;
    }
    else if(argMatches("bc1", argv[i]))
    {
      format = nvtt::Format_BC1;
    }
    else if(argMatches("bc1n", argv[i]))
    {
      format = nvtt::Format_BC1;
      bc1n   = true;
    }
    else if(argMatches("bc1a", argv[i]))
    {
      format = nvtt::Format_BC1a;
    }
    else if(argMatches("bc2", argv[i]))
    {
      format = nvtt::Format_BC2;
    }
    else if(argMatches("bc3", argv[i]))
    {
      format = nvtt::Format_BC3;
    }
    else if(argMatches("bc3n", argv[i]))
    {
      format = nvtt::Format_BC3n;
    }
    else if(argMatches("bc4", argv[i]))
    {
      format = nvtt::Format_BC4;
    }
    else if(argMatches("bc4s", argv[i]))
    {
      format = nvtt::Format_BC4S;
    }
    else if(argMatches("ati2", argv[i]))
    {
      format = nvtt::Format_ATI2;
    }
    else if(argMatches("bc5", argv[i]))
    {
      format = nvtt::Format_BC5;
    }
    else if(argMatches("bc5s", argv[i]))
    {
      format = nvtt::Format_BC5S;
    }
    else if(argMatches("bc6", argv[i]))
    {
      format = nvtt::Format_BC6U;
    }
    else if(argMatches("bc6s", argv[i]))
    {
      format = nvtt::Format_BC6S;
    }
    else if(argMatches("bc7", argv[i]))
    {
      format = nvtt::Format_BC7;
    }
    else if(argMatches("bc3_rgbm", argv[i]))
    {
      format = nvtt::Format_BC3_RGBM;
      rgbm   = true;
    }
    else if(argMatches("astc_ldr_4x4", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_4x4;
    }
    else if(argMatches("astc_ldr_5x4", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_5x4;
    }
    else if(argMatches("astc_ldr_5x5", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_5x5;
    }
    else if(argMatches("astc_ldr_6x5", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_6x5;
    }
    else if(argMatches("astc_ldr_6x6", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_6x6;
    }
    else if(argMatches("astc_ldr_8x5", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_8x5;
    }
    else if(argMatches("astc_ldr_8x6", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_8x6;
    }
    else if(argMatches("astc_ldr_8x8", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_8x8;
    }
    else if(argMatches("astc_ldr_10x5", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_10x5;
    }
    else if(argMatches("astc_ldr_10x6", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_10x6;
    }
    else if(argMatches("astc_ldr_10x8", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_10x8;
    }
    else if(argMatches("astc_ldr_10x10", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_10x10;
    }
    else if(argMatches("astc_ldr_12x10", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_12x10;
    }
    else if(argMatches("astc_ldr_12x12", argv[i]))
    {
      format = nvtt::Format_ASTC_LDR_12x12;
    }
    else if(argMatches("pause", argv[i]))
    {
      printf("Press ENTER\n");
      fflush(stdout);
      (void)getchar();
    }

    // Output options
    else if(argMatches("silent", argv[i]))
    {
      silent = true;
    }
    else if(argMatches("dds10", argv[i]))
    {
      dds10 = true;
    }
    else if(argMatches("profile", argv[i]))
    {
      profiling = true;
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
    printf("NVIDIA Texture Tools %u.%u.%u - Copyright NVIDIA Corporation 2007 - 2023\n", major, minor, rev);
  }

  if(inStr == nullptr)
  {
    printf("usage: nvtt_compress [options] infile(or dir) [outfile(or dir)]\n\n");

    printf("Input options:\n");
    printf("  -color          The input image is a color map (default).\n");
    printf("  -alpha          The input image has an alpha channel used for transparency.\n");
    printf("  -noalpha        The input image has no alpha channel used for transparency.\n");
    printf(
        "  -alpha_dithering  Enable alpha dithering. Can be followed by a number indicating the number of bits used in "
        "alpha dithering.\n");
    printf("  -normal         The input image is a normal map.\n");
    printf("  -tonormal       Convert input to normal map.\n");
    printf("  -clamp          Clamp wrapping mode (default).\n");
    printf("  -repeat         Repeat wrapping mode.\n");
    printf("  -nomips         Disable mipmap generation.\n");
    printf("  -max-mip-count  Maximum number of mipmaps. 0 and 1 are the same as -nomips; 2 generates the base mip and one more; and so on.\n");
    printf("  -min-mip-size   Minimum mipmap size; avoids generating mips whose width or height is smaller than this number. (default: 1)\n");
    printf("  -mipfilter      Mipmap filter. One of the following: box, triangle, kaiser.\n");
    printf("  -no-mip-gamma-correct  Do not convert to linear space when downsampling. (default: only for normal maps)\n");
    printf("  -rgbm           Transform input to RGBM.\n");
    printf("  -rangescale     Scale image to use entire color range.\n");
    printf("  -weight_r       Weight of R channel, default is 1.\n");
    printf("  -weight_g       Weight of G channel, default is 1.\n");
    printf("  -weight_b       Weight of B channel, default is 1.\n");
    printf(
        "  -weight_a       Weight of A channel, default is 1 when alpha is used, overwritten to 0 when alpha is not "
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
      p1.output = fs::path(p1.input).replace_extension("dds");
    }

    if(p1.output == p1.input)
    {
      p1.output.replace_extension("");
      p1.output += "_out.dds";
    }

    FileList.push_back(p1);
  }

  // Set input options.
  nvtt::WrapMode wrapMode = wrapRepeat ? nvtt::WrapMode_Repeat : nvtt::WrapMode_Clamp;

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

  // Automatically use dds10 if compressing to BC6 or BC7
  if(format == nvtt::Format_BC6U || format == nvtt::Format_BC6S || format == nvtt::Format_BC7
     || (format >= nvtt::Format_ASTC_LDR_4x4 && format <= nvtt::Format_ASTC_LDR_12x12))
  {
    dds10 = true;
  }

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

  MyErrorHandler errorHandler;

  /// ToDo
  unsigned long long batchSizeLimit = 104857600;

  unsigned long long                curBatchSize = 0;
  nvtt::BatchList                   batchList;
  std::vector<nvtt::Surface*>       SurfaceList;
  std::vector<nvtt::OutputOptions*> OutputOptionsList;
  std::vector<std::string>          compressingList;

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

      bool multiInputImage = false;

      nvtt::TextureType textureType(nvtt::TextureType_2D);


      if((maxMipCount > 1) && stringEqualsCaseInsensitive(input.extension().string(), ".dds")
         && format != nvtt::Format_BC3_RGBM && !rgbm && format != nvtt::Format_BC6U && format != nvtt::Format_BC6S)
      {
        if(images.loadDDS(inputStr.c_str()))
        {
          textureType = images.GetTextureType();

          image           = images.GetSurface(0, 0, SNorm);
          multiInputImage = (images.GetMipmapCount() > 1 || images.GetFaceCount() > 1);
        }
      }

      if(image.isNull())
      {
        if(!image.load(inputStr.c_str(), 0, SNorm, timingContext))
        {
          fprintf(stderr, "Error opening input file '%s'.\n", inputStr.c_str());
          return EXIT_FAILURE;
        }
        textureType = image.type();
      }

      nvtt::AlphaMode alphaMode = image.alphaMode();
      if(alpha_set)
        alphaMode = alpha ? nvtt::AlphaMode_Transparency : nvtt::AlphaMode_None;

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

        // To sRGB.
        if(mipGammaCorrect)
        {
          image.toSrgbUnclamped(timingContext);
        }

        if(format != nvtt::Format_BC3_RGBM)
        {
          alphaMode = nvtt::AlphaMode_None;
          image.toRGBM(1, 0.15f, timingContext);
        }
      }

      if(format == nvtt::Format_BC6U || format == nvtt::Format_BC6S)
        alphaMode = nvtt::AlphaMode_None;

      // Workaround for a bug in NVTT 3.2.0 and 3.2.1 where AlphaMode_None
      // and AlphaMode_Transparent are flipped when choosing a compressor.
      // See https://github.com/nvpro-samples/nvtt_samples/issues/3
      if(30200 <= NVTT_VERSION && NVTT_VERSION <= 30201)
      {
        if(alphaMode == nvtt::AlphaMode_Transparency)
        {
          alphaMode = nvtt::AlphaMode_None;
        }
        else
        {
          alphaMode = nvtt::AlphaMode_Transparency;
        }
      }

      image.setAlphaMode(alphaMode);

      const int faceCount = multiInputImage ? images.GetFaceCount() : 1;

      const int mip0Width   = image.width();
      const int mip0Height  = image.height();
      const int mip0Depth   = image.depth();
      int       mipmapCount = 1;
      while(mipmapCount < maxMipCount)
      {
        const int nextMip   = mipmapCount + 1;
        const int mipWidth  = std::max(1, mip0Width >> mipmapCount);
        const int mipHeight = std::max(1, mip0Height >> mipmapCount);
        if((mipWidth == 1 && mipHeight == 1) || (mipWidth < minMipSize) || (mipHeight < minMipSize))
        {
          break;
        }
        mipmapCount++;
      }

      nvtt::OutputOptions* outputOptions = new nvtt::OutputOptions;
      outputOptions->setErrorHandler(&errorHandler);
      outputOptions->setFileName(FileList[i].output.string().c_str());

      compressingList.push_back(FileList[i].input.string().c_str());

      if(dds10)
      {
        outputOptions->setContainer(nvtt::Container_DDS10);
      }

      //// compress procedure

      if(!context.outputHeader(textureType, mip0Width, mip0Height, mip0Depth, mipmapCount, normal || color2normal,
                               compressionOptions, *outputOptions))
      {
        fprintf(stderr, "Error writing file header %s.\n", FileList[i].output.string().c_str());
        delete outputOptions;
        continue;
      }

      OutputOptionsList.push_back(outputOptions);

      // Output images.
      for(int f = 0; f < faceCount; f++)
      {
        // Can we use the input SurfaceSet (true)? Or do we have to regenerate
        // mipmaps (false)?
        bool useSourceImages = true;

        if(f > 0)
          images.GetSurface(f, 0, image, SNorm);

        if(useCuda)
          image.ToGPU(timingContext);

        // To normal map.
        if(color2normal)
        {
          image.toGreyScale(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 0.0f, timingContext);
          image.toNormalMap(1.0f / 1.875f, 0.5f / 1.875f, 0.25f / 1.875f, 0.125f / 1.875f, timingContext);
        }

        // To linear space.
        if(!image.isNormalMap() && (mipmapCount > 1) && mipGammaCorrect)
        {
          image.toLinearFromSrgbUnclamped(timingContext);
        }

        nvtt::Surface tmp = image;
        if(!tmp.isNormalMap() && (mipmapCount > 1) && mipGammaCorrect)
        {
          tmp.toSrgbUnclamped(timingContext);
        }

        context.quantize(tmp, compressionOptions);
        nvtt::Surface* surf = new nvtt::Surface(tmp);
        SurfaceList.push_back(surf);
        batchList.Append(surf, f, 0, outputOptions);

        for(int m = 1; m < mipmapCount; m++)
        {
          if(useSourceImages)
          {
            if(!multiInputImage || m >= images.GetMipmapCount())
            {                           // One face is missing in this mipmap level.
              useSourceImages = false;  // If one level is missing, ignore the following source images.
            }
          }

          if(useSourceImages)
          {
            images.GetSurface(f, m, image, SNorm);
            if(useCuda)
              image.ToGPU(timingContext);
            // For already generated mipmaps, we need to convert to linear.
            if(!image.isNormalMap() && mipGammaCorrect)
            {
              image.toLinearFromSrgbUnclamped(timingContext);
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
            if(mipGammaCorrect)
            {
              tmp.toSrgbUnclamped(timingContext);
            }
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
    printf("\rTotal processing time: %.3f seconds\n\n", timeDiff(startTime, endTime));
  }

  if(profiling)
  {
    printf("\rDetailed profiling:\n");
    context.getTimingContext()->PrintRecords();
  }

  return EXIT_SUCCESS;
}
