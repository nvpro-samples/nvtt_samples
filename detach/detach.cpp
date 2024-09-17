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

#include <fileformats/nv_dds.h>

#include <array>
#include <filesystem>
#include <stdio.h>
#include <string>
#include <string.h>

void print_help()
{
  // clang-format off
  const char* helpText =
"NVIDIA Texture Tools - Copyright NVIDIA Corporation 2007 - 2024\n"
"\n"
"nvtt_detach - Splits a DDS file along layers, faces, and/or mips.\n"
"\n"
"Usage: nvtt_detach [--layers] [--faces] [--mips] input.dds\n"
"\n"
"Options:\n"
"  --layers: Splits array element 0, 1, ... into separate DDS files.\n"
"  --faces: Splits face 0, face 1, ... into separate DDS files.\n"
"  --mips: Splits mip 0, mip 1, ... into separate DDS files.\n"
"         nvtt_detach without any options is equivalent to nvtt_detach --mips.\n"
"  -h: Prints this text and exits.\n"
"\n"
"Examples:\n"
"  nvtt_detach wood.dds\n"
"    Splits the mips of wood.dds into separate DDS files named\n"
"    wood_M0.dds, wood_M1.dds, ...\n"
"  nvtt_detach --mips --faces wood.dds\n"
"    Splits along both mips and faces; output files will be named\n"
"    wood_F<face index>_M<mip index>.dds.\n";
"  nvtt_detach --layers --faces --mips wood.dds\n"
"    Splits along layers, mips, and faces; output files will be named\n"
"    wood_L<layer index>_F<face index>_M<mip index>.dds.\n";
  // clang-format on
  fprintf(stderr, "%s", helpText);
}

// Pseudo-iterator over 3D grids. This makes it easier to say "for each
// output DDS file: for each subresource within that DDS file..."
class Iterator3D
{
  std::array<uint32_t, 3> m_current{}, m_size{};

public:
  Iterator3D(const std::array<uint32_t, 3> size) { m_size = size; }

  Iterator3D& operator++()
  {
    // Increment m_current
    for(int i = 2; i >= 0; i--)
    {
      m_current[i]++;
      if(m_current[i] < m_size[i])
      {
        break;
      }
      else if(i != 0)
      {
        m_current[i] = 0;
      }
    }
    return *this;
  }

  operator bool() { return m_current[0] < m_size[0]; }

  const uint32_t& operator[](size_t n) { return m_current[n]; }
};

int main(int argc, char* argv[])
{
  const char* inPath      = nullptr;
  bool        splitLayers = false;
  bool        splitFaces  = false;
  bool        splitMips   = false;

  for(int argIndex = 1; argIndex < argc; argIndex++)
  {
    const char* arg = argv[argIndex];
    if(strcmp(arg, "-h") == 0)
    {
      print_help();
      return EXIT_SUCCESS;
    }
    else if(strcmp(arg, "--layers") == 0)
    {
      splitLayers = true;
    }
    else if(strcmp(arg, "--faces") == 0)
    {
      splitFaces = true;
    }
    else if(strcmp(arg, "--mips") == 0)
    {
      splitMips = true;
    }
    else
    {
      inPath = arg;
    }
  }

  if(nullptr == inPath)
  {
    fprintf(stderr, "Error: An input file must be specified.\n");
    print_help();
    return EXIT_FAILURE;
  }

  // Default split mode
  if(!splitLayers && !splitFaces && !splitMips)
  {
    splitMips = true;
  }

  const std::string basePath = std::filesystem::path(inPath).replace_extension("").string();

  // Read the base DDS file.
  nv_dds::Image              in;
  const nv_dds::ReadSettings readSettings{};  // Defaults are OK
  nv_dds::ErrorWithText      result = in.readFromFile(inPath, readSettings);
  if(result.has_value())
  {
    fprintf(stderr, "Error: Loading the file '%s' failed: %s\n", inPath, result.value().c_str());
    return EXIT_FAILURE;
  }

  // Order: layers, faces, mips
  enum ResourceRank
  {
    eLayer = 0,
    eFace,
    eMip
  };
  std::array<uint32_t, 3> outputFileGrid{};
  std::array<uint32_t, 3> subresourcesPerFile{};

  outputFileGrid[eLayer]      = (splitLayers ? in.getNumLayers() : 1);
  subresourcesPerFile[eLayer] = (splitLayers ? 1 : in.getNumLayers());

  outputFileGrid[eFace]      = (splitFaces ? in.getNumFaces() : 1);
  subresourcesPerFile[eFace] = (splitFaces ? 1 : in.getNumFaces());

  outputFileGrid[eMip]      = (splitMips ? in.getNumMips() : 1);
  subresourcesPerFile[eMip] = (splitMips ? 1 : in.getNumMips());

  for(Iterator3D outputFileIdx(outputFileGrid); outputFileIdx; ++outputFileIdx)
  {
    nv_dds::Image         out;
    nv_dds::ErrorWithText result =
        out.allocate(subresourcesPerFile[eMip], subresourcesPerFile[eLayer], subresourcesPerFile[eFace]);
    if(result.has_value())
    {
      fprintf(stderr, "Error: Allocating the DDS output failed: %s\n", result.value().c_str());
      return EXIT_FAILURE;
    }

    const uint32_t first_read_mip   = (splitMips ? outputFileIdx[eMip] : 0);
    const uint32_t first_read_layer = (splitLayers ? outputFileIdx[eLayer] : 0);
    const uint32_t first_read_face  = (splitFaces ? outputFileIdx[eFace] : 0);

    // Copy basic data from the input DDS
    out.mip0Width         = in.getWidth(first_read_mip);
    out.mip0Height        = in.getHeight(first_read_mip);
    out.mip0Depth         = in.getDepth(first_read_mip);
    out.dxgiFormat        = in.dxgiFormat;
    out.resourceDimension = in.resourceDimension;
    out.cubemapFaceFlags  = (splitFaces ? 0 : in.cubemapFaceFlags);
    out.alphaMode         = in.alphaMode;
    out.colorTransform    = in.colorTransform;
    out.isNormal          = in.isNormal;
    out.hasUserVersion    = in.hasUserVersion;
    out.userVersion       = in.userVersion;

    // Move subresources
    for(Iterator3D subresourceIdx(subresourcesPerFile); subresourceIdx; ++subresourceIdx)
    {
      const uint32_t in_mip   = first_read_mip + subresourceIdx[eMip];
      const uint32_t in_layer = first_read_layer + subresourceIdx[eLayer];
      const uint32_t in_face  = first_read_face + subresourceIdx[eFace];
      out.subresource(subresourceIdx[eMip], subresourceIdx[eLayer], subresourceIdx[eFace]) =
          std::move(in.subresource(in_mip, in_layer, in_face));
    }

    // Determine the output filename
    std::string outFilePath = basePath;
    if(splitLayers)
    {
      outFilePath += "_L" + std::to_string(outputFileIdx[eLayer]);
    }

    if(splitFaces)
    {
      outFilePath += "_F" + std::to_string(outputFileIdx[eFace]);
    }

    if(splitMips)
    {
      outFilePath += "_M" + std::to_string(outputFileIdx[eMip]);
    }

    outFilePath += ".dds";

    // Write the output
    nv_dds::WriteSettings writeSettings{};
    writeSettings.useDx10HeaderIfPossible = in.getFileInfo().hadDx10Extension;
    result                                = out.writeToFile(outFilePath.c_str(), writeSettings);
    if(result.has_value())
    {
      fprintf(stderr, "Writing file '%s' failed: %s\n", outFilePath.c_str(), result.value().c_str());
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
