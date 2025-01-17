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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANDOM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANDOM_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
enum RandomOptype { RANDOM_OP_NORMAL = 0, RANDOM_OP_UNIFORM_INT, RANDOM_OP_UNIFORM_REAL, RANDOM_OP_INVALID_TYPE = 255 };

const std::map<std::string, RandomOptype> kRandomOpTypeMap = {
  {"StandardNormal", RANDOM_OP_NORMAL}, {"UniformInt", RANDOM_OP_UNIFORM_INT}, {"UniformReal", RANDOM_OP_UNIFORM_REAL}};

class RandomCpuKernelMod : public NativeCpuKernelMod {
 public:
  RandomCpuKernelMod() = default;
  ~RandomCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  RandomOptype random_op_type_{RANDOM_OP_INVALID_TYPE};
  int seed_{0};
  int seed2_{0};
};

MS_REG_CPU_KERNEL(StandardNormal, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
                  RandomCpuKernelMod);
MS_REG_CPU_KERNEL(UniformInt,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  RandomCpuKernelMod)
MS_REG_CPU_KERNEL(UniformReal, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
                  RandomCpuKernelMod)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RANDOM_CPU_KERNEL_H_
