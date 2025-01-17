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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_TENSOR_DENSE_MATMUL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_TENSOR_DENSE_MATMUL_CPU_KERNEL_H_

#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename I, typename T>
class SparseTensorDenseMatmulCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseTensorDenseMatmulCpuKernelMod() = default;
  ~SparseTensorDenseMatmulCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  std::vector<size_t> output_shape_;
  std::vector<size_t> b_shape_;
  size_t output_size_{0};
  size_t values_size_{0};
  bool adj_st_{false};
  bool adj_dt_{false};
  enum input_list_ { INDICES, VALUES, SPARSE_SHAPE, DENSE };
};

MS_REG_CPU_KERNEL_T_S(SparseTensorDenseMatmul,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      SparseTensorDenseMatmulCpuKernelMod, int32_t, int32_t);

MS_REG_CPU_KERNEL_T_S(SparseTensorDenseMatmul,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      SparseTensorDenseMatmulCpuKernelMod, int32_t, int64_t);

MS_REG_CPU_KERNEL_T_S(SparseTensorDenseMatmul,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      SparseTensorDenseMatmulCpuKernelMod, int32_t, float);

MS_REG_CPU_KERNEL_T_S(SparseTensorDenseMatmul,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      SparseTensorDenseMatmulCpuKernelMod, int32_t, double);

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RMSPROP_CPU_KERNEL_H_
