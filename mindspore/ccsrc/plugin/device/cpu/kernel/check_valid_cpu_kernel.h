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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CHECK_VALID_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CHECK_VALID_CPU_KERNEL_H_

#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t COORDINATE = 4;
template <typename T>
class CheckValidCpuKernelMod : public NativeCpuKernelMod {
 public:
  CheckValidCpuKernelMod() = default;
  ~CheckValidCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParams(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  std::vector<size_t> anchor_box_shape_;
  std::vector<size_t> img_metas_shape_;
  std::vector<size_t> output_shape_;
};

MS_REG_CPU_KERNEL_T(
  CheckValid,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  CheckValidCpuKernelMod, float);

MS_REG_CPU_KERNEL_T(
  CheckValid,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  CheckValidCpuKernelMod, float16);

MS_REG_CPU_KERNEL_T(
  CheckValid, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
  CheckValidCpuKernelMod, int16_t);

MS_REG_CPU_KERNEL_T(
  CheckValid, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
  CheckValidCpuKernelMod, uint8_t);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CHECK_VALID_CPU_KERNEL_H_
