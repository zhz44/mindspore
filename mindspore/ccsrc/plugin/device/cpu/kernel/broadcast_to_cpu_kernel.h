/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BROADCAST_TO_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BROADCAST_TO_CPU_KERNEL_H_

#include <vector>
#include <memory>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"
#include "plugin/device/cpu/kernel/nnacl/base/broadcast_to.h"

namespace mindspore {
namespace kernel {
template <typename T>
class BroadcastToCpuKernelMod : public NativeCpuKernelMod {
 public:
  BroadcastToCpuKernelMod() = default;
  ~BroadcastToCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;
  void InitKernel(const CNodePtr &kernel_node) override;

  void CheckArgs();

 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  BroadcastShapeInfo shape_info_{};
};

MS_REG_CPU_KERNEL_T(BroadcastTo, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    BroadcastToCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(BroadcastTo, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    BroadcastToCpuKernelMod, int);
MS_REG_CPU_KERNEL_T(BroadcastTo, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                    BroadcastToCpuKernelMod, bool);
MS_REG_CPU_KERNEL_T(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  BroadcastToCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  BroadcastToCpuKernelMod, int);
MS_REG_CPU_KERNEL_T(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  BroadcastToCpuKernelMod, bool);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BROADCAST_TO_CPU_KERNEL_H_
