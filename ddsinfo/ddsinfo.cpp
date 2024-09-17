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

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
  if(argc != 2)
  {
    fprintf(stderr, "NVIDIA Texture Tools - Copyright NVIDIA Corporation 2007 - 2024\n\n");
    fprintf(stderr, "nvtt_ddsinfo - Prints information about a DDS file.\n");
    fprintf(stderr, "Usage: nvtt_ddsinfo ddsfile\n\n");
    return EXIT_FAILURE;
  }

  // Read the header of the DDS image -- no need to read beyond the header.
  nv_dds::Image               dds;
  const nv_dds::ReadSettings  settings{};  // Defaults are OK
  const nv_dds::ErrorWithText result = dds.readHeaderFromFile(argv[1], settings);
  if(result.has_value())
  {
    fprintf(stderr, "Loading the file '%s' failed: %s\n", argv[1], result.value().c_str());
    return EXIT_FAILURE;
  }

  printf("%s\n", dds.formatInfo().c_str());
  return EXIT_SUCCESS;
}