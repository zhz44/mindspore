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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SELECT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SELECT_CPU_KERNEL_H_

#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class SelectCpuKernelMod : public NativeCpuKernelMod {
 public:
  SelectCpuKernelMod() = default;
  ~SelectCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  void InitKernel(const CNodePtr &kernel_node) override;

 private:
  size_t element_num_{1};
};

MS_REG_CPU_KERNEL_T(Select,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeBool)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    SelectCpuKernelMod, float);

MS_REG_CPU_KERNEL_T(Select,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeBool)
                      .AddInputAttr(kNumberTypeFloat64)
                      .AddInputAttr(kNumberTypeFloat64)
                      .AddOutputAttr(kNumberTypeFloat64),
                    SelectCpuKernelMod, double);

MS_REG_CPU_KERNEL_T(Select,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeBool)
                      .AddInputAttr(kNumberTypeFloat16)
                      .AddInputAttr(kNumberTypeFloat16)
                      .AddOutputAttr(kNumberTypeFloat16),
                    SelectCpuKernelMod, float16);

MS_REG_CPU_KERNEL_T(Select,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeBool)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    SelectCpuKernelMod, int);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SELECT_CPU_KERNEL_H_
