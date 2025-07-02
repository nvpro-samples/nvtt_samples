/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
// Shows how to use NVTT 3's low-level GPUInputBuffer API to compress a texture
// directly from a CUDA buffer. Using this API allows the input and output to
// exist on the GPU, avoiding GPU-to-CPU and CPU-to-GPU copies.
//
// "-d [file]" can be passed on the command line to also decompress the image
// and save to the given file.
//
// --decompression-engine [cuda/cpu]: Decompresses using the given engine (default: cuda).


#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <ios>
#include <memory>
#include <nvpsystem.hpp>
#include <nvtt/nvtt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
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

// Simple CPU+GPU timer so we can time both NVTT and CUDA calls
struct Timer
{
  using Clock = std::chrono::high_resolution_clock;
  struct Event
  {
    std::string       name{};
    Clock::time_point cpuBegin{}, cpuEnd{};
    cudaEvent_t       gpuBegin{}, gpuEnd{};
  };

  // Creates a new event, records its CPU start time, and inserts a CUDA event
  // for its GPU start time.
  Event& begin(const char* name)
  {
    m_events.emplace_back();
    Event& e = m_events.back();
    e.name   = name;
    cudaEventCreate(&e.gpuBegin);
    cudaEventCreate(&e.gpuEnd);
    e.cpuBegin = Clock::now();
    cudaEventRecord(e.gpuBegin);
    return e;
  }

  // Records the CPU end time of an event, and inserts a CUDA event for its
  // GPU stop time.
  void end(Event& event)
  {
    event.cpuEnd = Clock::now();
    cudaEventRecord(event.gpuEnd);
  }

  // Times a lambda function.
  template <typename F>
  decltype(auto) time(const char* name, F&& lambda)
  {
    Event& e      = begin(name);
    auto   result = lambda();
    end(e);
    return result;
  }

  // Prints timing information for all events.
  // Make sure to synchronize before calling this function.
  void report()
  {
    printf("CPU (seconds)\tGPU (seconds)\tEvent name\n");
    for(const Event& event : m_events)
    {
      const double cpuSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(event.cpuEnd - event.cpuBegin).count();
      float gpuMilliseconds = -1.f;
      cudaEventElapsedTime(&gpuMilliseconds, event.gpuBegin, event.gpuEnd);
      printf("%f\t%f\t%s\n", cpuSeconds, gpuMilliseconds / 1000.f, event.name.c_str());
    }
  }

private:
  std::vector<Event> m_events;
};

// Quantizes from a float to an 8-bit UNORM.
// This rounds to the nearest value in {0, 1/255, ..., 254/255, 1},
// so that converting back to a float by dividing by 255 gives correct rounding.
__device__ unsigned char quantizeToU8(float x)
{
  return static_cast<unsigned char>(__fmaf_rd(__saturatef(x), 255.f, .5f));
}

// Copies data from a texture to a linear buffer.
__global__ void copyTextureToBufferU4(cudaTextureObject_t tex, uchar4* output, uint32_t width, uint32_t height)
{
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height)
  {
    float4 col = tex2D<float4>(tex, x, y);
    // Quantize and swizzle from RGBA32_SFLOAT to BGRA8_UNORM
    uchar4 u8x4{quantizeToU8(col.z), quantizeToU8(col.y), quantizeToU8(col.x), quantizeToU8(col.w)};
    output[y * width + x] = u8x4;
  }
}

int main(int argc, char** argv)
{
  const char* decompressToFile = nullptr;
  enum class DecompressionEngine
  {
    eCuda,
    eCpu
  } decompressionEngine = DecompressionEngine::eCuda;

  // Parse arguments
  for(int argIndex = 1; argIndex < argc; argIndex++)
  {
    const char* arg = argv[argIndex];
    if(strcmp(arg, "-h") == 0)
    {
      printf(
          "nvtt_cuda_input - Shows how to use the GPUInputBufferAPI to "
          "compress a texture directly from a CUDA buffer.\n"
          "usage: nvtt_cuda_input [options]\n"
          "  -d [file]: Also decompress the image and save to the given file.\n"
          "  --decompression-engine [cuda/cpu]: Decompresses using the given engine (default: cuda)\n"
          "  -h: Display this help text and exit.\n");

      return EXIT_SUCCESS;
    }
    else if(strcmp(arg, "-d") == 0)
    {
      argIndex++;
      if(argIndex == argc)
      {
        printf("Error: -d must be followed by a filename.\n");
        return EXIT_FAILURE;
      }
      decompressToFile = argv[argIndex];
    }
    else if(strcmp(arg, "--decompression-engine") == 0)
    {
      argIndex++;
      if(argIndex == argc)
      {
        printf("Error: --decompression-engine must be followed by a filename.\n");
        return EXIT_FAILURE;
      }
      arg = argv[argIndex];
      if(strcmp(arg, "cuda") == 0)
      {
        decompressionEngine = DecompressionEngine::eCuda;
      }
      else if(strcmp(arg, "cpu") == 0)
      {
        decompressionEngine = DecompressionEngine::eCpu;
      }
      else
      {
        printf("Error: Unrecognized --decompression-engine: %s\n", arg);
        return EXIT_FAILURE;
      }
    }
    else
    {
      printf("Error: Unrecognized option: %s\n", arg);
      return EXIT_FAILURE;
    }
  }

  Timer timer;
#define TIME(expr) timer.time(#expr, [&] { return expr; })

  if(!nvtt::isCudaSupported())
  {
    printf(
        "Error: Attempting to run the nvtt_cuda_input sample on a system that doesn't support the CUDA requirements "
        "needed. This could mean that there's no GPU that supports CUDA, that the graphics driver should be updated to "
        "support the version of CUDA NVTT 3 uses, or that all GPUs had compute capability less than 3.0.\n");
    return EXIT_FAILURE;
  }

  // Format settings
  const char*                  formatName = "BC7";
  const nvtt::Format           format     = nvtt::Format::Format_BC7;
  const cudaResourceViewFormat cudaFormat = cudaResViewFormatUnsignedBlockCompressed7;
  const int                    tileSize[2]{4, 4};

  // Create the input. We might use GPUInputBuffer to compress something we
  // created on the GPU, which usually means we'd know the format and the size.
  // Here, we'll approximate this by compressing cuda_input.raw, which is a
  // 704x618 8-bit sRGB image with channels stored in interleaved
  // [blue, green, red, alpha] order.
  nvtt::RefImage inputImage{};
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
  std::vector<char>              rawData;
  for(const std::string& path : searchPaths)
  {
    try
    {
      std::ifstream        file(path + "/cuda_input.raw", std::ios::binary | std::ios::ate);
      const std::streampos lengthSigned = file.tellg();
      if(lengthSigned > 0)
      {
        const size_t length = static_cast<size_t>(lengthSigned);
        rawData.resize(length);
        file.seekg(file.beg);
        file.read(rawData.data(), length);
        break;
      }
    }
    catch(const std::exception& /* unused */)
    {
    }
  }
  if(rawData.empty())
  {
    printf("Error: Could not locate cuda_input.raw!\n");
    return EXIT_FAILURE;
  }

  // Upload the raw data to the GPU
  void*        d_inputData    = nullptr;
  const size_t inputSizeBytes = static_cast<size_t>(inputImage.width) * static_cast<size_t>(inputImage.height)
                                * static_cast<size_t>(inputImage.num_channels) * sizeof(uint8_t);
  CUDA_CHECK(TIME(cudaMalloc(&d_inputData, inputSizeBytes)));
  CUDA_CHECK(TIME(cudaMalloc(&d_inputData, inputSizeBytes)));
  CUDA_CHECK(TIME(cudaMemcpy(d_inputData, reinterpret_cast<const void*>(rawData.data()), inputSizeBytes, cudaMemcpyHostToDevice)));
  inputImage.data = d_inputData;

  // Now that CUDA's set up, make sure NVTT uses the same GPU as the one we
  // uploaded the image to.
  nvtt::useCurrentDevice();

  // Construct a GPUInputBuffer; this can refer to multiple images to compress
  // them all at once, but here we'll only compress one. In particular, it
  // segments the input into tiles for processing, which must be the block size
  // of the input format (4x4 for BC formats, a variable size for ASTC formats).
  std::unique_ptr<nvtt::GPUInputBuffer> gpuInput =
      TIME(std::make_unique<nvtt::GPUInputBuffer>(&inputImage,               // Array of RefImages
                                                  nvtt::ValueType::UINT8,    // The type of the elements of the image
                                                  1,                         // Number of RefImages
                                                  tileSize[0], tileSize[1],  // Tile dimensions
                                                  1.0F, 1.0F, 1.0F, 1.0F  // Weights for prioritizing different channels for quality.
                                                  ));

  // Get the size of the compressed data.
  int compressedSizeBytes{};
  int tileSizeBytes{};
  {
    nvtt::Context            context;
    nvtt::CompressionOptions options;
    options.setFormat(format);

    compressedSizeBytes =
        context.estimateSize(inputImage.width, inputImage.height, inputImage.depth, 1 /* number of mipmaps */, options);
    tileSizeBytes = context.estimateSize(1, 1, 1, 1, options);
  }

  printf("Compressing cuda_input.raw to %s format...\n", formatName);

  // NVTT 3.2 unified all its low-level compression APIs using EncodeSettings,
  // which also makes some things more concise.
  const nvtt::EncodeSettings encodeSettings = nvtt::EncodeSettings().SetFormat(format).SetOutputToGPUMem(true);
  void*                      d_compressedData{};  // GPU (device) data
  // For decompression:
  cudaArray_t         d_array{};          // Optimal-tiled data
  cudaTextureObject_t d_texture{};        // Texture object wrapper
  uchar4*             hd_decompressed{};  // Page-locked memory allocated with cudaMallocHost

  CUDA_CHECK(TIME(cudaMalloc(&d_compressedData, compressedSizeBytes)));

  if(!TIME(nvtt::nvtt_encode(*gpuInput, d_compressedData, encodeSettings)))
  {
    fprintf(stderr, "Encoding failed!\n");
    goto CleanUp;
  }

  {
    // Write the output data to a file to give an example of what this creates.
    // This won't be readable unless we have something like a DDS header or have
    // knowledge of the format and image dimensions.
    std::vector<char> h_compressedData(compressedSizeBytes);
    CUDA_CHECK(cudaMemcpy(h_compressedData.data(), d_compressedData, compressedSizeBytes, cudaMemcpyDeviceToHost));
    std::ofstream outFile("out.raw", std::ios::binary);
    outFile.write(h_compressedData.data(), compressedSizeBytes);
    outFile.close();
  }

  printf("Compressed data in %s format without a header has been written to out.raw.\n", formatName);

  //-----------------------------------------------------------------------------------------------------------------//
  // Decompression: with and without hardware acceleration                                                           //
  //-----------------------------------------------------------------------------------------------------------------//
  if(decompressToFile)
  {
    nvtt::Surface decompressed;
    if(decompressionEngine == DecompressionEngine::eCuda)
    {
      // Decompression using GPU hardware.
      //
      // Interpreting the data and blitting it can be a bit tricky. Since we
      // want to use the GPU's texturing hardware, we need to get to a
      // CUDA texture object; then we can use a kernel to access it and use
      // pinned memory to copy directly to host memory.
      //
      // The CUDA documentation only specifies block-compressed behavior for
      // textures when the underlying resource is a CUDA array or mipmapped
      // array. So we'll allocate memory for one of those and copy. Attempting
      // to use linear arrays or 2D pitched arrays will result in
      // cudaErrorInvalidValue.
      //
      // Conceptually, our array here is an array of tiles, rather than of
      // pixels. Each element uses either 64 (BC1, BC4) or 128 bits.
      const size_t widthTiles  = (inputImage.width + tileSize[0] - 1) / tileSize[0];
      const size_t heightTiles = (inputImage.height + tileSize[1] - 1) / tileSize[1];
      {
        cudaChannelFormatDesc formatDesc{};
        formatDesc.x = 32;
        formatDesc.y = 32;
        formatDesc.f = cudaChannelFormatKindUnsigned;
        if(tileSizeBytes == 16)
        {
          formatDesc.z = formatDesc.w = 32;
        }
        CUDA_CHECK(TIME(cudaMallocArray(&d_array, &formatDesc, widthTiles, heightTiles)));
      }

      // Copy from linear GPU memory to the array.
      CUDA_CHECK(TIME(cudaMemcpy2DToArray(d_array, 0, 0,                                 // dst, wOffset, hOffset
                                          d_compressedData, widthTiles * tileSizeBytes,  // src, spitch
                                          widthTiles * tileSizeBytes, heightTiles,  // width in bytes, height in rows
                                          cudaMemcpyDeviceToDevice)));

      cudaResourceDesc resourceDesc{};
      resourceDesc.resType         = cudaResourceTypeArray;
      resourceDesc.res.array.array = d_array;
      cudaTextureDesc textureDesc{};
      textureDesc.addressMode[0]   = cudaAddressModeClamp;
      textureDesc.addressMode[1]   = cudaAddressModeClamp;
      textureDesc.addressMode[2]   = cudaAddressModeClamp;
      textureDesc.filterMode       = cudaFilterModePoint;      // This allows us to avoid adding .5 to coordinates
      textureDesc.readMode         = cudaReadModeElementType;  // Read back values directly
      textureDesc.normalizedCoords = false;
      cudaResourceViewDesc viewDesc{};
      viewDesc.format = cudaFormat;  // View the data in the array as this format
      // When we create a BC7 resource view, the width and height must be an
      // exact multiple of the tile size, or CUDA will reject it.
      // For images with size not divisible by the tile size, this means that
      // our view can contain a few extra pixels on the side -- it does not
      // crop to the true image bounds for us. We'll do this cropping in the
      // kernel below.
      viewDesc.width  = static_cast<size_t>(widthTiles * tileSize[0]);
      viewDesc.height = static_cast<size_t>(heightTiles * tileSize[1]);
      // This must match the array's depth exactly. Since 2D arrays have a
      // depth of 0 (not 1), this is 0.
      viewDesc.depth = 0;
      CUDA_CHECK(TIME(cudaCreateTextureObject(&d_texture, &resourceDesc, &textureDesc, &viewDesc)));

      // Allocate output storage
      // This allocates host memory (RAM) that's accessible on the GPU. This
      // means that when our kernel below writes to it, it'll also transfer
      // the data at the same time. This allows us to skip calls to cudaMalloc
      // and to malloc(), and is efficient since we write to each location only
      // once and adjacent threads write to similar locations -- it saves
      // about 1ms (about 20%) on @nbickford's test system.
      //
      // Texturing hardware outputs RGBA32_FLOAT; we'll quantize on the GPU to
      // BGRA8_UNORM, since this reduces the amount of data we need to transfer
      // over PCIe (which is the bottleneck).
      const size_t decompressedSizeBytes =
          static_cast<size_t>(inputImage.width) * static_cast<size_t>(inputImage.height) * sizeof(uchar4);
      CUDA_CHECK(TIME(cudaMallocHost(&hd_decompressed, decompressedSizeBytes)));

      const dim3 blockSize(16, 16);
      const dim3 numBlocks((uint32_t(inputImage.width) + blockSize.x - 1) / blockSize.x,
                           (uint32_t(inputImage.height) + blockSize.y - 1) / blockSize.y);
      timer.time("copyTextureToBufferU4", [&]() {
        copyTextureToBufferU4<<<numBlocks, blockSize>>>(d_texture, hd_decompressed, static_cast<uint32_t>(inputImage.width),
                                                        static_cast<uint32_t>(inputImage.height));
        return true;  // Dummy return value to make timer.time work
      });

      // Since the device -> host copy isn't explicit, it's wise to sync here
      // to wait for the kernel (and thus the transfer) to finish:
      CUDA_CHECK(TIME(cudaDeviceSynchronize()));

      // Copy to an NVTT surface, just to visualize the result.
      TIME(decompressed.setImage(nvtt::InputFormat_BGRA_8UB, inputImage.width, inputImage.height, 1, hd_decompressed));
    }
    else
    {
      // Decompression on the CPU.
      // Our compressed data is on the GPU. We did a GPU->CPU copy for debug
      // purposes above; we'll do that again to display the time it takes.
      std::vector<char> h_compressedData;
      timer.time("h_compressedData.resize(...)", [&]() {
        h_compressedData.resize(compressedSizeBytes);
        return true;
      });
      CUDA_CHECK(TIME(cudaMemcpy(h_compressedData.data(), d_compressedData, compressedSizeBytes, cudaMemcpyDeviceToHost)));
      TIME(decompressed.setImage2D(format, inputImage.width, inputImage.height, h_compressedData.data()));
    }

    const bool saveOK = TIME(decompressed.save(decompressToFile, true));
    if(saveOK)
    {
      printf("Successfully saved decompressed image to %s.\n", decompressToFile);
    }
    else
    {
      printf("Failed to save decompressed image to %s.\n", decompressToFile);
    }
  }

  printf("\nTiming info:\n");
  timer.report();

CleanUp:
  CUDA_CHECK(cudaFreeHost(hd_decompressed));
  CUDA_CHECK(cudaDestroyTextureObject(d_texture));
  CUDA_CHECK(cudaFreeArray(d_array));
  CUDA_CHECK(cudaFree(d_inputData));
  CUDA_CHECK(cudaFree(d_compressedData));
  return EXIT_SUCCESS;
}
