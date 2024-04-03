/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <string>

#define nvDebugBreak() __debugbreak();

inline bool stringEqualsCaseInsensitive(const std::string& a, const std::string& b)
{
  const size_t len = a.size();
  if(len != b.size())
  {
    return false;
  }
  for(size_t i = 0; i < len; i++)
  {
    if(tolower(a[i]) != tolower(b[i]))
    {
      return false;
    }
  }
  return true;
}

// Returns the number of seconds from time point `a` to time point `b`.
template <class TimePoint>
inline float timeDiff(const TimePoint& a, const TimePoint& b)
{
  return std::chrono::duration_cast<std::chrono::duration<float>>(b - a).count();
}
