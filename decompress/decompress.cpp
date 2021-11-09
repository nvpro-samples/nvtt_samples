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
#include <filesystem>
#include <nvtt/nvtt.h>
#include <string.h>
#include <string>

namespace fs = std::filesystem;
using Clock  = std::chrono::high_resolution_clock;

int main(int argc, char* argv[])
{

  bool forcenormal = false;
  bool mipmaps     = false;
  bool faces       = false;
  bool savePNG     = false;
  bool rgbm        = false;
  bool histogram   = false;

  fs::path input;
  fs::path output;

  // Parse arguments.
  for(int i = 1; i < argc; i++)
  {
    if(strcmp("-forcenormal", argv[i]) == 0)
    {
      forcenormal = true;
    }
    else if(strcmp("-mipmaps", argv[i]) == 0)
    {
      mipmaps = true;
    }
    else if(strcmp("-rgbm", argv[i]) == 0)
    {
      rgbm = true;
    }
    else if(strcmp("-faces", argv[i]) == 0)
    {
      faces = true;
    }
    else if(strcmp("-histogram", argv[i]) == 0)
    {
      histogram = true;
    }
    else if(strcmp("-format", argv[i]) == 0)
    {
      if(i + 1 == argc)
        break;
      i++;

      // !!!UNDONE: Support at least one HDR output format

#ifdef HAVE_PNG
      if(strcmp("png", argv[i]) == 0)
        savePNG = true;
      else
#endif
          if(strcmp("tga", argv[i]) == 0)
        savePNG = false;
      else
      {
        fprintf(stderr, "Unsupported output format '%s', defaulting to 'tga'.\n", argv[i]);
        savePNG = false;
      }
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
        output = input;
      }

      break;
    }
    else
    {
      printf("Warning: unrecognized option \"%s\"\n", argv[i]);
    }
  }

  printf("NVIDIA Texture Tools - Copyright NVIDIA Corporation 2015 - 2021\n\n");

  if(input.empty())
  {
    printf("usage: nvtt_decompress [options] infile.dds [outfile]\n\n");

    printf("Note: the .tga or .png extension is forced on outfile\n\n");

    printf("Input options:\n");
    printf("  -forcenormal      The input image is a normal map.\n");
    printf("  -mipmaps          Decompress all mipmaps.\n");
    printf("  -faces            Decompress all faces.\n");
    printf("  -histogram        Output histogram.\n");
    printf("  -format <format>  Output format ('tga' or 'png').\n");

    return 1;
  }


  if(histogram)
  {
    nvtt::Surface img;
    if(!img.load(input.string().c_str()))
    {
      fprintf(stderr, "The file '%s' is not a valid DDS file.\n", input.string().c_str());
      return 1;
    }

    float exposure = 2.2f;
    float scale    = 1.0f / exposure;
    img.scaleBias(0, scale, 0);
    img.scaleBias(1, scale, 0);
    img.scaleBias(2, scale, 0);

    //img.toneMap(nvtt::ToneMapper_Reindhart, NULL);
    //img.toSrgb();
    img.toGamma(2.2f);

    nvtt::Surface hist = nvtt::histogram(img, 3 * 512, 128);

    // Resize for pretier histograms.
    hist.resize(512, 128, 1, nvtt::ResizeFilter_Box);

    fs::path name = output.replace_extension(".histogram");
    name += fs::path(savePNG ? ".png" : ".tga");

    hist.save(name.string().c_str());
  }
  else
  {

    // Load surface set
    nvtt::SurfaceSet images;
    if(!images.loadDDS(input.string().c_str(), forcenormal))
    {
      fprintf(stderr, "Error opening input file '%s'.\n", input.string().c_str());
      return 1;
    }

    uint32_t faceCount   = images.GetFaceCount();
    uint32_t mipmapCount = images.GetMipmapCount();

    const auto startTime = Clock::now();

    // apply arguments
    if(!faces)
    {
      faceCount = 1;
    }
    if(!mipmaps)
    {
      mipmapCount = 1;
    }

    fs::path name;

    // extract faces and mipmaps
    for(uint32_t f = 0; f < faceCount; f++)
    {
      for(uint32_t m = 0; m < mipmapCount; m++)
      {
        // set output filename, if we are doing faces and/or mipmaps
        name = output.replace_extension("");
        if(faces)
          name += "_face" + std::to_string(f);
        if(mipmaps)
          name += "_mipmap" + std::to_string(m);
        name += (savePNG ? ".png" : ".tga");

        if(!images.saveImage(name.string().c_str(), f, m))
        {
          fprintf(stderr, "Error opening '%s' for writting\n", name.string().c_str());
          return 1;
        }
      }
    }

    const auto endTime = Clock::now();
    printf("\rtime taken: %.3f seconds\n", timeDiff(startTime, endTime));
  }

  return 0;
}
