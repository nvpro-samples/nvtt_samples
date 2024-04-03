/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
// Shows how to use some features of NVTT 3's C wrapper. This allows NVTT 3
// to be used from any programming language with support for C bindings, as
// well as from C++ compilers with different name mangling schemes.
//
// All C functions and types have C++ equivalents, so please refer to the
// C++ documentation for their full specifications.
//
// This app uses a few NVTT 3 C wrapper functions to show the general approach.
// It loads an image, converts it to a normal map, and then compresses and
// saves it in a few different ways.

#include <errno.h>
#include <nvtt/nvtt_wrapper.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_help_text()
{
  printf(
      "nvtt_c_wrapper_demo - Shows how to use the NVTT 3 C API to perform tasks similar to the C++ API. Compresses the "
      "given file using BC7 to c_wrapper_demo_input.dds, converts the file to a normal map, saves it to "
      "c_wrapper_demo_out_normal.tga, and saves compressed BC5 data to c_wrapper_demo_out_normal.dds.\n");
  printf("usage: nvtt_c_wrapper_demo [filename] [options]\n");
  printf("  -h: Display this help text.\n");
}

int main(int argc, char** argv)
{
  char*                   filename = NULL;
  int                     i        = 0;
  NvttTimingContext*      tc;       // This can be set to NULL to avoid timing operations.
  NvttSurface*            surface;  // An uncompressed floating-point RGBA image.
  NvttContext*            context;  // An NVTT compression context. One of the high-level APIs for compressing textures.
  NvttCompressionOptions* compressionOptions;  // Includes what compression format to use and other encoding options.
  NvttOutputOptions*      outputOptions;       // Stores information such as a file or custom handler to write to.
  int                     numMipmaps;
  char*                   rawOutput = NULL;
  int                     exitCode  = EXIT_SUCCESS;

  if(argc <= 1)
  {
    print_help_text();
    return EXIT_FAILURE;
  }

  // Parse arguments
  for(i = 1; i < argc; i++)
  {
    if(strcmp(argv[i], "-h") == 0)
    {
      print_help_text();
      return EXIT_SUCCESS;
    }
    else
    {
      filename = argv[i];
    }
  }

  // Make sure we got a file name
  if(filename == NULL)
  {
    print_help_text();
    return EXIT_FAILURE;
  }

  // Set up the timing context. Measure all functions.
  tc = nvttCreateTimingContext(3);

  // Load the source image into a floating-point RGBA image.
  surface = nvttCreateSurface();  // Equivalent to nvtt::Surface surface;

  // Equivalent to if(!surface.load(filename, &sourceHadAlpha))
  {
    NvttBoolean sourceHadAlpha;
    if(nvttSurfaceLoad(surface, filename, &sourceHadAlpha, NVTT_False, NULL) == NVTT_False)
    {
      printf("Loading the file at %s into an NVTT Surface failed!\n", filename);
      exitCode = EXIT_FAILURE;
      goto CleanUp;
    }
  }

  //---------------------------------------------------------------------------
  // Now, let's create a mipmapped BC7-compressed DDS file.

  // Create the compression context.
  context = nvttCreateContext();  // Equivalent to nvtt::Context context;

  // Enable CUDA compression, so that GPUs that support it will use GPU
  // acceleration for compression, with a fallback on other GPUs for CPU
  // compression.
  nvttSetContextCudaAcceleration(context, NVTT_True);  // Equivalent to context.setCudaAcceleration(true);

  // Specify what compression settings to use. In our case the only default
  // to change is that we want to compress to BC7.
  compressionOptions = nvttCreateCompressionOptions();                   // nvtt::CompressionOptions compressionOptions;
  nvttSetCompressionOptionsFormat(compressionOptions, NVTT_Format_BC7);  // compressionOptions.setFormat(nvtt::Format_BC7);

  // Specify how to output the compressed data. Here, we say to write to a file.
  // We could also use a custom output handler here instead.
  outputOptions = nvttCreateOutputOptions();                                // nvtt::OutputOptions outputOptions;
  nvttSetOutputOptionsFileName(outputOptions, "c_wrapper_demo_input.dds");  // outputOptions.setFileName("c_wrapper_demo_input.dds");

  // Compute the number of mips.
  numMipmaps = nvttSurfaceCountMipmaps(surface, 1);  // numMipmaps = image.countMipmaps();

  // Write the DDS header. Since this uses the BC7 format, this will
  // automatically use the DX10 DDS extension.
  // Equivalent to if(!context.outputHeader(surface, numMipmaps, compressionOptions, outputOptions))
  if(nvttContextOutputHeader(context, surface, numMipmaps, compressionOptions, outputOptions) == NVTT_False)
  {
    printf("Writing the DDS header failed!\n");
    exitCode = EXIT_FAILURE;
    goto CleanUp;
  }

  // Create mips, compress them, and convert them to the file. See the mipmap
  // sample for why we do the color space conversions here.
  {
    // In the C++ API, it's usually better to use the copy assignment
    // operator instead, nvtt::Surface workingSurface = surface; this does a
    // shallow copy, and then NVTT does a deep copy only when needed.
    // Here, we know we're going to modify the contents, and so we do a clone.
    NvttSurface* workingSurface = nvttSurfaceClone(surface);
    int          mip;

    for(mip = 0; mip < numMipmaps; mip++)
    {
      // Compress this image and write its data.
      // Equivalent to if(!context.compress(workingSurface, 0, mip, compressionOptions, outputOptions))
      if(nvttContextCompress(context, workingSurface, 0 /* face */, mip, compressionOptions, outputOptions) == NVTT_False)
      {
        printf("Compressing and writing the DDS file failed!\n");
        exitCode = EXIT_FAILURE;
        goto CleanUp;
      }

      if(mip == numMipmaps - 1)
        break;

      // Prepare the next mip by converting to linear premultiplied alpha,
      // resizing, and then converting back to use unpremultiplied sRGB:

      nvttSurfaceToLinearFromSrgb(workingSurface, tc);  // workingSurface.toLinearFromSrgb(tc);
      nvttSurfacePremultiplyAlpha(workingSurface, tc);  // workingSurface.premultiplyAlpha(tc);
      nvttSurfaceBuildNextMipmapDefaults(workingSurface, NVTT_MipmapFilter_Box, 1, tc);  // workingSurface.buildNextMipmap(nvtt::MipmapFilter_Box, 1, tc);
      nvttSurfaceDemultiplyAlpha(workingSurface, 1e-12f, tc);  // workingSurface.demultiplyAlpha(1e-12f, tc);
      nvttSurfaceToSrgb(workingSurface, tc);                   // workingSurface.toSrgb(tc);
    }

    nvttDestroySurface(workingSurface);
  }

  //---------------------------------------------------------------------------
  // Now let's take the original surface, turn it into a normal map using a
  // weighted average of its RGB channels, and save it to a .tga file.

  nvttSurfaceToGreyScale(surface, 2.0f, 4.0f, 1.0f, 0.0f, tc);  // surface.toGreyScale(2.0f, 4.0f, 1.0f, 0.0f, tc);
  // Copy red to the alpha channel since normal mapping uses that:
  nvttSurfaceCopyChannel(surface, surface, 0, 3, tc);  // surface.copyChannel(0, 3, tc);
  // Scale the heightmap down:
  nvttSurfaceScaleBias(surface, 3, 0.2f, 0.0f, tc);             // surface.scaleBias(0.1f, 0.0f, tc);
  nvttSurfaceToNormalMap(surface, 2.0f, 1.0f, 1.0f, 1.0f, tc);  // surface.toNormalMap(4.0f, 2.0f, 1.0f, 0.5f, tc);
  // We'll use BC5U, so all values must be positive:
  nvttSurfacePackNormals(surface, 0.5f, 0.5f, tc);  // surface.packNormals(0.5f, 0.5f, tc);

  // if(!surface.save("c_wrapper_demo_out_normal.tga", false, false, tc))
  if(nvttSurfaceSave(surface, "c_wrapper_demo_out_normal.tga", NVTT_False, NVTT_False, tc) == NVTT_False)
  {
    printf("Saving the normal map to c_wrapper_demo_out_normal.tga failed!\n");
    return EXIT_FAILURE;
  }

  //---------------------------------------------------------------------------
  // Finally, let's compress the red and green channels using BC5 and the
  // low-level API.
  {
    NvttRefImage        refImage;
    unsigned            num_tiles;
    NvttCPUInputBuffer* cpuInputBuffer = NULL;
    int                 rawOutputSize;
    FILE*               outfile = NULL;

    refImage.data               = nvttSurfaceData(surface);    // refImage.data = surface.data();
    refImage.width              = nvttSurfaceWidth(surface);   // refImage.width = surface.width();
    refImage.height             = nvttSurfaceHeight(surface);  //refImage.height = surface.height();
    refImage.depth              = nvttSurfaceDepth(surface);   // refImage.depth = surface.depth();
    refImage.num_channels       = 4;
    refImage.channel_swizzle[0] = NVTT_ChannelOrder_Red;
    refImage.channel_swizzle[1] = NVTT_ChannelOrder_Green;
    refImage.channel_swizzle[2] = NVTT_ChannelOrder_Blue;   // This doesn't matter for BC5, which is a 2-channel format
    refImage.channel_swizzle[3] = NVTT_ChannelOrder_Alpha;  // This doesn't matter for BC5, which is a 2-channel format
    refImage.channel_interleave = NVTT_False;

    cpuInputBuffer = nvttCreateCPUInputBuffer(&refImage,               // Array of images
                                              NVTT_ValueType_FLOAT32,  // Input data type - NVTT 3 Surfaces are flaoting-point images
                                              1,                       // Number of images
                                              4, 4,                    // BC5 width and height
                                              1.0f, 1.0f, 1.0f, 1.0f,  // Error metric channel weights
                                              tc, &num_tiles);

    // If we know that BC5 uses 16 bytes per 4x4 tile, we can use
    // output = malloc(num_tiles * 16);
    // However, it's less memorization to use NVTT's functions:
    nvttResetCompressionOptions(compressionOptions);
    nvttSetCompressionOptionsFormat(compressionOptions, NVTT_Format_BC5);
    rawOutputSize = nvttContextEstimateSizeData(context, refImage.width, refImage.height, refImage.depth, 1, compressionOptions);
    if(rawOutputSize <= 0)
    {
      printf("The compressed output size was invalid (zero or negative) - is the image 0x0?\n");
      exitCode = EXIT_FAILURE;
      goto CleanUp;
    }
    rawOutput = malloc(rawOutputSize);
    if(rawOutput == NULL)
    {
      printf("Failed to allocate compressed output buffer!\n");
      exitCode = EXIT_FAILURE;
      goto CleanUp;
    }

    // Now compress the data!
    nvttEncodeBC5CPU(cpuInputBuffer,         // Input
                     NVTT_False,             // Whether to use the slow, high-quality mode
                     rawOutput,              // Output
                     nvttIsCudaSupported(),  // Whether to use GPU acceleration
                     NVTT_False,             // Whether the output resides on the GPU
                     tc);                    // Timing context

    // Now, we could output the raw data, but for the purposes of this demo,
    // we'll use the high-level API to write a DDS header, then append the BC5
    // data to the file. That way the output is a valid DDS file instead of a
    // raw buffer.
#ifdef _MSC_VER
    const errno_t openErrno = fopen_s(&outfile, "c_wrapper_demo_out_normal.dds", "wb");
#else
    outfile             = fopen("c_wrapper_demo_out_normal.dds", "wb");
    const int openErrno = errno;
#endif
    if(outfile == NULL)
    {
      printf("Could not open c_wrapper_demo_out_normal.dds for writing! (Errno %i)\n", openErrno);
      exitCode = EXIT_FAILURE;
      goto CleanUp;
    }
#ifdef _MSC_VER
    // On Windows, the first piece of code that writes to a C file handle also
    // creates its buffer. In this case, that's normally NVTT. This leads to
    // an issue where the memory allocator linked inside NVTT created the file
    // handle's buffer, but external code needs to deallocate that file handle.
    // To work around it, allocate the buffer here; then we know how to
    // deallocate it.
    const int setvbufReturnCode = setvbuf(outfile, NULL, _IOFBF, 1U << 16);
    if(setvbufReturnCode == -1)
    {
      printf("Could not allocate internal buffer for file handle! (Errno %i)\n", errno);
      exitCode = EXIT_FAILURE;
      goto CleanUp;
    }
#endif

    nvttResetOutputOptions(outputOptions);
    nvttSetOutputOptionsFileHandle(outputOptions, outfile);
    nvttContextOutputHeaderData(context, NVTT_TextureType_2D, refImage.width, refImage.height, refImage.depth, 1,
                                NVTT_True, compressionOptions, outputOptions);
    fwrite(rawOutput, 1, (size_t)(rawOutputSize), outfile);
    fclose(outfile);
  }

  // Print out timing information.
  nvttTimingContextPrintRecords(tc);

  printf("Demo finished.\n");

  // Finally, clean up.
CleanUp:
  if(rawOutput)
    free(rawOutput);
  if(outputOptions)
    nvttDestroyOutputOptions(outputOptions);
  if(compressionOptions)
    nvttDestroyCompressionOptions(compressionOptions);
  if(context)
    nvttDestroyContext(context);
  if(surface)
    nvttDestroySurface(surface);
  if(tc)
    nvttDestroyTimingContext(tc);

  return exitCode;
}