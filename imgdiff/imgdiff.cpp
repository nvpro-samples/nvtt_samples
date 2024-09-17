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
 * SPDX-FileCopyrightText: Copyright (c) 2007-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.h"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <math.h>
#include <nvtt/nvtt.h>
#include <string.h>

namespace fs = std::filesystem;
using Clock  = std::chrono::high_resolution_clock;

int main(int argc, char* argv[])
{
  bool compareNormal = false;
  bool compareAlpha  = false;
  bool rangescale    = false;

  fs::path input0, input1, output;

  // Parse arguments.
  for(int i = 1; i < argc; i++)
  {
    // Input options.
    if(strcmp("-normal", argv[i]) == 0)
    {
      compareNormal = true;
    }
    else if(strcmp("-alpha", argv[i]) == 0)
    {
      compareAlpha = true;
    }
    else if(strcmp("-rangescale", argv[i]) == 0)
    {
      rangescale = true;
    }
    else if(argv[i][0] != '-')
    {
      input0 = argv[i];

      if(i + 1 < argc && argv[i + 1][0] != '-')
      {
        input1 = argv[i + 1];
      }

      if(i + 2 < argc && argv[i + 2][0] != '-')
      {
        output = argv[i + 2];
      }

      break;
    }
    else
    {
      printf("Warning: unrecognized option \"%s\"\n", argv[i]);
    }
  }

  if(input0.empty() || input1.empty())
  {
    printf("NVIDIA Texture Tools - Copyright NVIDIA Corporation 2007 - 2024\n\n");

    printf("nvtt_imgdiff - Measures difference between two images and optionally writes a difference image.\n");
    printf("usage: nvtt_imgdiff [options] original_file updated_file [output]\n\n");

    printf("Diff options:\n");
    printf("  -normal       Compare images as if they were normal maps.\n");
    printf("  -alpha        Compare alpha weighted images.\n");
    printf("  -rangescale   Scale second image based on range of first one.\n");
    printf("Diff output:\n");
    printf("  If specified, the difference between the two images will be written to [output].\n");
    printf("  This can be e.g. a .tga file.\n");

    return 1;
  }

  nvtt::Surface image0, image1;

  if(!image0.load(input0.string().c_str()))
  {
    printf("Error loading %s.", input0.string().c_str());
    return 1;
  }
  if(!image1.load(input1.string().c_str()))
  {
    printf("Error loading %s.", input1.string().c_str());
    return 1;
  }

  if(image0.width() != image1.width())
  {
    printf("Error: First image had a different width (%d) than the second image (%d).\n", image0.width(), image1.width());
    return 1;
  }

  if(image0.height() != image1.height())
  {
    printf("Error: First image had a different height (%d) than the second image (%d).\n", image0.height(), image1.height());
    return 1;
  }

  if(compareNormal)
  {
    image0.setNormalMap(true);
    image1.setNormalMap(true);
  }
  if(compareAlpha)
  {
    image0.setAlphaMode(nvtt::AlphaMode_Transparency);
  }

  // Do some transforms based on the naming convention of the file.
  if(input1.string().find("rgbm") != std::string::npos)
  {
    image1.fromRGBM(1.0f, 0.15f);
    image1.toLinear(2.2f);

    image1.copyChannel(image0, 3);  // Copy alpha channel from source.
    image1.setAlphaMode(nvtt::AlphaMode_Transparency);

    rangescale = true;
  }

  if(input1.string().find("bc3n") != std::string::npos)
  {
    // Undo how BC3N maps (r, g, b, a) to (1, g, 0, r):
    image1.copyChannel(image1, 3, 0);  // Copy channel 3 to channel 0
    image1.copyChannel(image0, 2);     // Copy the blue channel from source
    image1.copyChannel(image0, 3);     // Copy the alpha channel from source
  }

  if(input1.string().find("bc6") != std::string::npos)
  {
    // @@ Do any transform that we may have done before compression.

    image1.copyChannel(image0, 3);  // Copy alpha channel from source.
    image1.setAlphaMode(nvtt::AlphaMode_Transparency);
  }


  // Scale second image to range of the first one.
  if(rangescale)
  {
    float min_color[3], max_color[3];
    image0.range(0, &min_color[0], &max_color[0]);
    image0.range(1, &min_color[1], &max_color[1]);
    image0.range(2, &min_color[2], &max_color[2]);
    float color_range = std::max({max_color[0], max_color[1], max_color[2]});

    const float max_color_range = 16.0f;
    if(color_range > max_color_range)
      color_range = max_color_range;

    for(int i = 0; i < 3; i++)
    {
      image1.scaleBias(i, color_range, 0.0f);
    }
  }

  if(image0.alphaMode() == nvtt::AlphaMode_Transparency)
  {
    printf("RGB RMSE measurements weighted by reference alpha channel.\n");
  }

  const float rmse = nvtt::rmsError(image0, image1);
  printf("RGB RMSE: %f\n", rmse);
  printf("RGB MSE:  %f\n", rmse * rmse);
  if(rmse == 0.0f)
  {
    printf("RGB PSNR: Identical images.\n");
  }
  else
  {
    printf("RGB PSNR: %f\n", 20.0f * log10f(1.0f / rmse));
  }

  const float rmsa = nvtt::rmsAlphaError(image0, image1);
  printf("Alpha RMSE: %f\n", rmsa);

  const float rmse0 = nvtt::rmsToneMappedError(image0, image1, 2.2f);
  const float rmse1 = nvtt::rmsToneMappedError(image0, image1, 1.0f);
  const float rmse2 = nvtt::rmsToneMappedError(image0, image1, 0.22f);

  printf("Tone mapped RMSE @ exposure of 2.2f:  %.5f\n", rmse0);
  printf("Tone mapped RMSE @ exposure of 1.0f:  %.5f\n", rmse1);
  printf("Tone mapped RMSE @ exposure of 0.22f: %.5f\n", rmse2);
  printf("Average: %.5f\n", (rmse0 + rmse1 + rmse2) / 3);

  if(compareNormal)
  {
    float ae = nvtt::angularError(image0, image1);
    printf("RMS angular error = %f\n", ae);
  }

  if(!output.empty())
  {
    // Write image difference.
    nvtt::Surface diff = image0;
    for(int c = 0; c < 4; c++)
    {
      diff.addChannel(image1, c, c, -1.0f);
      diff.abs(c);
    }
    bool saveSuccess = diff.save(output.string().c_str(), true, false);
    if(!saveSuccess)
    {
      printf("Writing the difference image to %s failed.\n", output.string().c_str());
      return 1;
    }
  }

  return 0;
}
