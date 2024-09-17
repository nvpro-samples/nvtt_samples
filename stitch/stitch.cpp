/*
 * Copyright (c) 2007-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2007-2024, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#define NV_DDS_UTILITY_VALUES
#include <fileformats/nv_dds.h>
#include <fileformats/texture_formats.h>

#include <filesystem>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <utility>

namespace fs = std::filesystem;

void printHelp()
{
  // clang-format off
  const char* helpText =
"NVIDIA Texture Tools - Copyright NVIDIA Corporation 2007-2024\n"
"\n"
"nvtt_stitch - Combines DDS files for individual layers, faces, and/or mips\n"
"into a single DDS file.\n"
"\n"
"Usage: nvtt_stitch [--layers] [--faces] [--mips] base_filename.dds\n"
"Creates base_filename.dds; looks for files named base_filename<suffix>.dds as\n"
"described below."
"\n"
"Options:\n"
"  --layers: Combines multiple array elements together.\n"
"            <suffix> starts with '_L<layer index>'.\n"
"  --faces: Combines multiple faces together.\n"
"           <suffix> is followed by '_F<face index>'.\n"
"  --mips: Combines multiple mips together.\n"
"          <suffix> ends with '_M<mip index>' or '_<mip index>'.\n"
"          nvtt_stitch without any options is equivalent to nvtt_stitch --mips.\n"
"  -o, --output: Writes to a different file name."
"\n"
"Examples:\n"
"  nvtt_stitch wood.dds"
"    Creates wood.dds from files named wood_0.dds, wood_1.dds, ... or\n"
"    wood_M0.dds, wood_M1.dds, ..., each of which contains 1 mip.\n"
"  nvtt_stitch --mips --faces wood.dds\n"
"    Creates wood.dds from files named wood_F<face index>_M<mip index>.dds or\n"
"    wood_F<face index>_<mip index>.dds.\n"
"  nvtt_stitch --layers --faces --mips wood.dds\n"
"    Creates wood.dds from files named\n"
"    wood_L<layer index>_F<face index>_M<mip index>.dds or\n"
"    wood_L<layer index>_F<face index>_<mip index>.dds.\n";
  // clang-format on
  fprintf(stderr, "%s", helpText);
}

int main(int argc, char* argv[])
{
  const char* basePathWithExtension = nullptr;
  const char* outPath               = nullptr;
  bool        stitchLayers          = false;
  bool        stitchFaces           = false;
  bool        stitchMips            = false;

  for(int argIndex = 1; argIndex < argc; argIndex++)
  {
    const char* arg = argv[argIndex];
    if(strcmp(arg, "-h") == 0)
    {
      printHelp();
      return EXIT_SUCCESS;
    }
    else if(strcmp(arg, "--layers") == 0)
    {
      stitchLayers = true;
    }
    else if(strcmp(arg, "--faces") == 0)
    {
      stitchFaces = true;
    }
    else if(strcmp(arg, "--mips") == 0)
    {
      stitchMips = true;
    }
    else if(strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0)
    {
      if(argIndex == argc - 1)
      {
        fprintf(stderr, "Error: %s must be followed by a file name.", arg);
        return EXIT_FAILURE;
      }
      argIndex++;
      outPath = argv[argIndex];
    }
    else
    {
      basePathWithExtension = arg;
    }
  }

  if(nullptr == basePathWithExtension)
  {
    fprintf(stderr, "Error: A base file must be specified.\n");
    printHelp();
    return EXIT_FAILURE;
  }

  // Out path defaults to the base path with the extension
  if(nullptr == outPath)
  {
    outPath = basePathWithExtension;
  }

  // Default split mode
  if(!stitchLayers && !stitchFaces && !stitchMips)
  {
    stitchMips = true;
  }

  const std::string basePath = fs::path(basePathWithExtension).replace_extension("").string();

  // Order: layers, faces, mips
  enum ResourceRank
  {
    eLayer = 0,
    eFace,
    eMip
  };

  // Finds the file name for a given resource index; returns an empty string
  // if it couldn't find any files matching that specification.
  auto findFile = [&](uint32_t layer, uint32_t face, uint32_t mip) -> std::string {
    std::string searchPath = basePath;
    if(stitchLayers)
    {
      searchPath += "_L" + std::to_string(layer);
    }

    if(stitchFaces)
    {
      searchPath += "_F" + std::to_string(face);
    }

    if(stitchMips)
    {
      const std::string mipStr = std::to_string(mip);
      // Try the _M name first
      const std::string nameWithM = searchPath + "_M" + mipStr + ".dds";
      if(fs::exists(fs::path(nameWithM)))
      {
        return nameWithM;
      }
      // If we couldn't find that, try it without M.
      searchPath += "_" + mipStr;
    }

    searchPath += ".dds";
    if(fs::exists(fs::path(searchPath)))
    {
      return searchPath;
    }
    return "";
  };

  // Make sure at least one file exists.
  if(findFile(0, 0, 0).empty())
  {
    fprintf(stderr,
            "Error: Could not find any input files!\n"
            "Did they follow the naming convention in the help text? "
            "(You can see it by running nvtt_stitch -h)\n");
    std::string examplePath = basePath;
    if(stitchLayers)
      examplePath += "_L0";
    if(stitchFaces)
      examplePath += "_F0";

    if(stitchMips)
    {
      fprintf(stderr, "For instance, there should be a file named '%s_M0.dds' or '%s_0.dds'.\n", examplePath.c_str(),
              examplePath.c_str());
    }
    else
    {
      fprintf(stderr, "For instance, there should be a file named '%s.dds'.\n", examplePath.c_str());
    }
    return EXIT_FAILURE;
  }

  // Determine the number of layers, faces, and mips.
  uint32_t layerFiles = 1;
  uint32_t faceFiles  = 1;
  uint32_t mipFiles   = 1;
  if(stitchLayers)
  {
    // While the next layer exists, increment the number of layers.
    while(!findFile(layerFiles, 0, 0).empty())
    {
      layerFiles++;
    }
  }

  if(stitchFaces)
  {
    while(!findFile(0, faceFiles, 0).empty())
    {
      faceFiles++;
    }
  }

  if(stitchMips)
  {
    while(!findFile(0, 0, mipFiles).empty())
    {
      mipFiles++;
    }
  }

  // Build the output DDS file
  nv_dds::Image         out;
  nv_dds::ErrorWithText result;
  uint32_t              layersPerFile = 1;
  uint32_t              facesPerFile  = 1;
  uint32_t              mipsPerFile   = 1;
  for(uint32_t layerFile = 0; layerFile < layerFiles; layerFile++)
  {
    for(uint32_t faceFile = 0; faceFile < faceFiles; faceFile++)
    {
      for(uint32_t mipFile = 0; mipFile < mipFiles; mipFile++)
      {
        const std::string inPath = findFile(layerFile, faceFile, mipFile);
        if(inPath.empty())
        {
          fprintf(stderr,
                  "Error: Could not find the file for layer %u, face %u, and mip %u! "
                  "This program guessed by searching files that there should have been "
                  "%u layers, %u faces, and %u mips. Are some DDS files missing, "
                  "or some extra files left over from an earlier process using the same base name?\n",
                  layerFile, faceFile, mipFile,  //
                  layerFiles, faceFiles, mipFiles);
          return EXIT_FAILURE;
        }

        nv_dds::Image              in;
        const nv_dds::ReadSettings readSettings{};  // Defaults are OK
        result = in.readFromFile(inPath.c_str(), readSettings);
        if(result.has_value())
        {
          fprintf(stderr, "Error: Loading the file '%s' failed: %s\n", inPath.c_str(), result.value().c_str());
          return EXIT_FAILURE;
        }

        // Which subresource of the output does (0,0,0) of this file correspond to?
        // For the first file, we won't know {layers, faces, mips}PerFile yet,
        // but this is OK since {layer, face, mip}File will be 0.
        const uint32_t firstReadLayer = layersPerFile * layerFile;
        const uint32_t firstReadFace  = facesPerFile * faceFile;
        const uint32_t firstReadMip   = mipsPerFile * mipFile;

        // For the first file, initialize the output.
        if(layerFile == 0 && faceFile == 0 && mipFile == 0)
        {
          out.mip0Width  = in.mip0Width;
          out.mip0Height = in.mip0Height;
          out.mip0Depth  = in.mip0Depth;

          out.dxgiFormat        = in.dxgiFormat;
          out.resourceDimension = in.resourceDimension;

          out.alphaMode      = in.alphaMode;
          out.colorTransform = in.colorTransform;
          out.isNormal       = in.isNormal;
          out.hasUserVersion = in.hasUserVersion;
          out.userVersion    = in.userVersion;

          layersPerFile = in.getNumLayers();
          facesPerFile  = in.getNumFaces();
          mipsPerFile   = in.getNumMips();

          size_t totalLayers = 0;
          size_t totalFaces  = 0;
          size_t totalMips   = 0;
          if(!checked_math::mul2(layersPerFile, layerFiles, totalLayers) || totalLayers > std::numeric_limits<uint32_t>::max())
          {
            fprintf(stderr,
                    "Error: The number of layers per file (%u) times "
                    "the number of files with different layers (%u) would have "
                    "overflowed a 32-bit unsigned integer!\n",
                    layersPerFile, layerFiles);
            return EXIT_FAILURE;
          }

          if(!checked_math::mul2(facesPerFile, faceFiles, totalFaces) || totalFaces > std::numeric_limits<uint32_t>::max())
          {
            fprintf(stderr,
                    "Error: The number of faces per file (%u) times "
                    "the number of files with different faces (%u) would have "
                    "overflowed a 32-bit unsigned integer!\n",
                    facesPerFile, faceFiles);
            return EXIT_FAILURE;
          }

          if(!checked_math::mul2(mipsPerFile, mipFiles, totalMips) || totalMips > std::numeric_limits<uint32_t>::max())
          {
            fprintf(stderr,
                    "Error: The number of mips per file (%u) times "
                    "the number of files with different mips (%u) would have "
                    "overflowed a 32-bit unsigned integer!\n",
                    mipsPerFile, mipFiles);
            return EXIT_FAILURE;
          }

          if(totalFaces > 6)
          {
            fprintf(stderr,
                    "Error: There were %u faces provided in the input "
                    "files (%u faces per file times %u files with different "
                    "faces), but DDS files can only have up to 6 faces.\n",
                    static_cast<uint32_t>(totalFaces), facesPerFile, faceFiles);
            return EXIT_FAILURE;
          }

          // Some bit trickery to set cubemapFaceFlags from the number of faces,
          // assigning them in order (since we don't have information as to
          // if we want a different set of faces in incomplete cubemaps, this
          // should be fine for now):
          out.cubemapFaceFlags = nv_dds::DDSCAPS2_CUBEMAP_POSITIVEX * ((1 << totalFaces) - 1);

          result = out.allocate(static_cast<uint32_t>(totalMips), static_cast<uint32_t>(totalLayers),
                                static_cast<uint32_t>(totalFaces));
          if(result.has_value())
          {
            fprintf(stderr, "Error: Allocating the DDS output failed: %s\n", result.value().c_str());
            return EXIT_FAILURE;
          }
        }
        else
        {
          // For subsequent files, check that they have the same parameters
          // and a size that matches what we'd expect.
          const uint32_t expectedWidth  = out.getWidth(firstReadMip);
          const uint32_t expectedHeight = out.getHeight(firstReadMip);
          const uint32_t expectedDepth  = out.getDepth(firstReadMip);
          if(expectedWidth != in.getWidth(0)       //
             || expectedHeight != in.getHeight(0)  //
             || expectedDepth != in.getDepth(0))
          {
            fprintf(stderr,
                    "Error: File '%s' should have had a size of %u x %u x %u "
                    "(since the first file had size %u x %u x %u and this file "
                    "should have contained data starting at mip %u), but it "
                    "actually had size %u x %u x %u.\n",
                    inPath.c_str(), expectedWidth, expectedHeight, expectedDepth,  //
                    out.getWidth(0), out.getHeight(0), out.getDepth(0),            //
                    firstReadMip,                                                  //
                    in.getWidth(0), in.getHeight(0), in.getDepth(0));
            return EXIT_FAILURE;
          }

          if(layersPerFile != in.getNumLayers()   //
             || facesPerFile != in.getNumFaces()  //
             || mipsPerFile != in.getNumMips())
          {
            fprintf(stderr,
                    "Error: File '%s' had %u layers, %u faces, and %u mips, "
                    "but it should have had the same numbers as the first file: "
                    "%u layers, %u faces, and %u mips.\n",
                    inPath.c_str(), in.getNumLayers(), in.getNumFaces(), in.getNumMips(),  //
                    layersPerFile, facesPerFile, mipsPerFile);
            return EXIT_FAILURE;
          }

          if(out.dxgiFormat != in.dxgiFormat)
          {
            fprintf(stderr,
                    "Error: File '%s' had a different format (%s) than "
                    "the first file (%s). All files must use the same format.\n",
                    inPath.c_str(), texture_formats::getDXGIFormatName(out.dxgiFormat),
                    texture_formats::getDXGIFormatName(in.dxgiFormat));
            return EXIT_FAILURE;
          }

          // We don't really mind of resourceDimension is wrong.

          if(out.alphaMode != in.alphaMode)
          {
            fprintf(stderr,
                    "Error: File '%s' had a different alpha mode (%u == %s) "
                    "than the first file (%u == %s). All files must use the same alpha mode.\n",
                    inPath.c_str(), out.alphaMode, nv_dds::getAlphaModeString(out.alphaMode), in.alphaMode,
                    nv_dds::getAlphaModeString(in.alphaMode));
            return EXIT_FAILURE;
          }

          if(out.colorTransform != in.colorTransform)
          {
            fprintf(stderr,
                    "Error: File '%s' had a different color transform (%s) "
                    "than the first file (%s). All files must use the same color transform.\n",
                    inPath.c_str(), nv_dds::getColorTransformString(out.colorTransform),
                    nv_dds::getColorTransformString(in.colorTransform));
            return EXIT_FAILURE;
          }

          // We don't really mind if the isNormal or user version flags are
          // set differently.
        }

        // Move this file's data to the output:
        for(uint32_t layerIn = 0; layerIn < layersPerFile; layerIn++)
        {
          for(uint32_t faceIn = 0; faceIn < facesPerFile; faceIn++)
          {
            for(uint32_t mipIn = 0; mipIn < mipsPerFile; mipIn++)
            {
              out.subresource(firstReadMip + mipIn, firstReadLayer + layerIn, firstReadFace + faceIn) =
                  std::move(in.subresource(mipIn, layerIn, faceIn));
            }
          }
        }
        // End of work for this file
      }
    }
  }

  // Write the output file:
  const nv_dds::WriteSettings writeSettings{};  // Defaults are fine
  result = out.writeToFile(outPath, writeSettings);
  if(result.has_value())
  {
    fprintf(stderr, "Error: Saving the file '%s' failed: %s\n", outPath, result.value().c_str());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
