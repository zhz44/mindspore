/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNPACK_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNPACK_CPU_KERNEL_H_

#include <algorithm>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"
#include "nnacl/base/unstack_base.h"

namespace mindspore {
namespace kernel {
template <typename T>
class UnpackCpuKernelMod : public NativeCpuKernelMod {
 public:
  UnpackCpuKernelMod() = default;
  ~UnpackCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  void InitInputOutputSize(const CNodePtr &kernel_node) override;

  UnstackParameter unstack_param_{};
  size_t output_num_{0};
};
MS_REG_CPU_KERNEL_T(Unstack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                    UnpackCpuKernelMod, int8_t);
MS_REG_CPU_KERNEL_T(Unstack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                    UnpackCpuKernelMod, int16_t);
MS_REG_CPU_KERNEL_T(Unstack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    UnpackCpuKernelMod, int);
MS_REG_CPU_KERNEL_T(Unstack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    UnpackCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(Unstack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                    UnpackCpuKernelMod, bool);
MS_REG_CPU_KERNEL_T(Unstack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                    UnpackCpuKernelMod, uint8_t);
MS_REG_CPU_KERNEL_T(Unstack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                    UnpackCpuKernelMod, uint16_t);
MS_REG_CPU_KERNEL_T(Unstack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                    UnpackCpuKernelMod, uint32_t);
MS_REG_CPU_KERNEL_T(Unstack,
                    KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                    UnpackCpuKernelMod, uint64_t);
MS_REG_CPU_KERNEL_T(
  Unstack, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  UnpackCpuKernelMod, float16);
MS_REG_CPU_KERNEL_T(
  Unstack, KernelAttr().SetAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  UnpackCpuKernelMod, float);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNPACK_CPU_KERNEL_H_
