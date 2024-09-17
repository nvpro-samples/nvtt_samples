/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
// Shows how to use NVTT 3's low-level GPUInputBuffer API to compress a texture
// directly from a CUDA buffer. Using this API allows the input and output to
// exist on the GPU, avoiding GPU-to-CPU and CPU-to-GPU copies.
// "-c" can be passed on the command line to output compressed data to CPU
// memory instead of to GPU memory.
// "-t" can be passed on the command line to print out timing statistics, at
// the expense of more cudaDeviceSynchronize() calls.

#include <cuda_runtime.h>
#include <fstream>
#include <ios>
#include <memory>
#include <nvh/fileoperations.hpp>
#include <nvpsystem.hpp>
#include <nvtt/nvtt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>

bool cudaCheck(cudaError_t result, const char* call, const char* functionName)
{
  if(result != cudaSuccess)
  {
    printf("CUDA call %s in %s failed with result code %d (%s)!\n", call, functionName, result, cudaGetErrorName(result));
    cudaDeviceReset();
    return false;
  }
  return true;
}

#define CUDA_CHECK(result)                                                                                             \
  if(!cudaCheck((result), #result, __func__))                                                                          \
  {                                                                                                                    \
    return EXIT_FAILURE;                                                                                               \
  }

int main(int argc, char** argv)
{
  bool                                 outputToGPU = true;
  std::unique_ptr<nvtt::TimingContext> timingContext;

  // Parse arguments
  for(int argIndex = 1; argIndex < argc; argIndex++)
  {
    if(strcmp(argv[argIndex], "-h") == 0)
    {
      printf(
          "nvtt_cuda_input - Shows how to use the GPUInputBufferAPI to "
          "compress a texture directly from a CUDA buffer.\n");
      printf("usage: nvtt_cuda_input [options]\n");
      printf("  -c: Output compressed data to CPU memory instead of to GPU memory.\n");
      printf(
          "  -t: Use nvtt::TimingContext to print out performance statistics, at the expense of more "
          "cudaDeviceSynchronize() calls.\n");
      printf("  -h: Display this help text.\n");

      return EXIT_SUCCESS;
    }
    else if(strcmp(argv[argIndex], "-c") == 0)
    {
      outputToGPU = false;
    }
    else if(strcmp(argv[argIndex], "-t") == 0)
    {
      timingContext = std::make_unique<nvtt::TimingContext>(4);
    }
  }

  if(!nvtt::isCudaSupported())
  {
    printf(
        "Error: Attempting to run the nvtt_cuda_input sample on a system that doesn't support the CUDA requirements "
        "needed. This could mean that there's no GPU that supports CUDA, that the graphics driver should be updated to "
        "support the version of CUDA NVTT 3 uses, or that all GPUs had compute capability less than 3.0.\n");
    return EXIT_FAILURE;
  }

  // Create the input. We might use GPUInputBuffer to compress something we
  // created on the GPU, which usually means we'd know the format and the size.
  // Here, we'll approximate this by compressing cuda_input.raw, which is a
  // 704x618 8-bit sRGB image with channels stored in interleaved
  // [blue, green, red, alpha] order.
  nvtt::RefImage inputImage;
  inputImage.width              = 704;
  inputImage.height             = 618;
  inputImage.depth              = 1;
  inputImage.num_channels       = 4;
  inputImage.channel_swizzle[0] = nvtt::Blue;
  inputImage.channel_swizzle[1] = nvtt::Green;
  inputImage.channel_swizzle[2] = nvtt::Red;
  inputImage.channel_swizzle[3] = nvtt::Alpha;
  inputImage.channel_interleave = true;

  const std::vector<std::string> searchPaths = {NVPSystem::exePath() + PROJECT_RELDIRECTORY,
                                                NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
                                                NVPSystem::exePath() + PROJECT_RELDIRECTORY "../..",
                                                NVPSystem::exePath() + PROJECT_NAME};
  const std::string              rawData     = nvh::loadFile("cuda_input.raw", true, searchPaths, true);
  if(rawData.empty())
  {
    printf("Error: Could not locate cuda_input.raw!\n");
    return EXIT_FAILURE;
  }

  // Upload the raw data to the GPU
  void*        d_inputData    = nullptr;
  const size_t inputSizeBytes = static_cast<size_t>(inputImage.width) * static_cast<size_t>(inputImage.height)
                                * static_cast<size_t>(inputImage.num_channels) * sizeof(uint8_t);
  CUDA_CHECK(cudaMalloc(&d_inputData, inputSizeBytes));
  CUDA_CHECK(cudaMemcpy(d_inputData, reinterpret_cast<const void*>(rawData.data()), inputSizeBytes, cudaMemcpyHostToDevice));
  inputImage.data = d_inputData;

  // Now that CUDA's set up, make sure NVTT uses the same GPU as the one we
  // uploaded the image to.
  nvtt::useCurrentDevice();

  // Construct a GPUInputBuffer; this can refer to multiple images to compress
  // them all at once, but here we'll only compress one. In particular, it
  // segments the input into tiles for processing, which must be the block size
  // of the input format (4x4 for BC formats, a variable size for ASTC formats).
  const nvtt::GPUInputBuffer gpuInput(&inputImage,             // Array of RefImages
                                      nvtt::ValueType::UINT8,  // The type of the elements of the image
                                      1,                       // Number of RefImages
                                      4, 4,                    // Tile dimensions of BC7
                                      1.0F, 1.0F, 1.0F, 1.0F  // Weights for prioritizing different channels for quality.
  );

  // Get the size of the compressed data. Here, we'll use BC7, which must match
  // the low-level encoding function we'll use.
  std::streamsize outputSizeBytes{};
  {
    nvtt::Context            context;
    nvtt::CompressionOptions options;
    options.setFormat(nvtt::Format_BC7);

    outputSizeBytes =
        context.estimateSize(inputImage.width, inputImage.height, inputImage.depth, 1 /* number of mipmaps */, options);
  }

  printf("Compressing cuda_input.raw to BC7 format...\n");

  // NVTT 3.2 unified all its low-level compression APIs using EncodeSettings,
  // which also makes some things more concise.
  nvtt::EncodeSettings encodeSettings =
      nvtt::EncodeSettings().SetFormat(nvtt::Format_BC7).SetTimingContext(timingContext.get()).SetOutputToGPUMem(outputToGPU);

  if(outputToGPU)
  {
    // Compress the data using both GPU input and output!
    void* d_outputData;
    CUDA_CHECK(cudaMalloc(&d_outputData, outputSizeBytes));
    if(!nvtt::nvtt_encode(gpuInput, d_outputData, encodeSettings))
    {
      fprintf(stderr, "Encoding failed!\n");
    }

    // Write the output data to a file to give an example of what this creates.
    // This won't be readable unless we have something like a DDS header or have
    // knowledge of the format and image dimensions.
    std::vector<char> outputData(outputSizeBytes);
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_outputData, outputSizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_outputData));

    std::ofstream outFile("out.raw", std::ios::binary);
    outFile.write(outputData.data(), outputSizeBytes);
    outFile.close();
  }
  else
  {
    // Compress the data using the GPU input and CPU output!
    std::vector<char> outputData(outputSizeBytes);
    if(!nvtt::nvtt_encode(gpuInput, outputData.data(), encodeSettings))
    {
      fprintf(stderr, "Encoding failed!\n");
    }

    // Write the output data to a file to give an example of what this creates.
    // This won't be readable unless we have something like a DDS header or have
    // knowledge of the format and image dimensions.
    std::ofstream outFile("out.raw", std::ios::binary);
    outFile.write(outputData.data(), outputSizeBytes);
    outFile.close();
  }

  CUDA_CHECK(cudaFree(d_inputData));

  printf("Done. Compressed data in BC7 format without a header has been written to out.raw.\n");

  if(timingContext)
  {
    printf("Timing statistics: \n");
    timingContext->PrintRecords();
  }

  return EXIT_SUCCESS;
}