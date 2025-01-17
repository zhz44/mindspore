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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_GPU_KERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unique_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class UniqueGpuKernelMod : public NativeGpuKernelMod {
 public:
  UniqueGpuKernelMod() { ResetResource(); }
  ~UniqueGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    S *input_index = GetDeviceAddress<S>(workspace, 0);
    S *sorted_index = GetDeviceAddress<S>(workspace, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);
    S *index = GetDeviceAddress<S>(outputs, 1);
    stream_ptr_ = stream_ptr;
    post_output_size_ = CalUnique(input, num_elements_, input_index, sorted_index, output, index,
                                  reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    std::vector<size_t> shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (auto x : shape) {
      num_elements_ *= x;
    }
    input_size_ = num_elements_ * sizeof(T);
    output_size_ = input_size_;
    workspace_size_ = num_elements_ * sizeof(S);
    InitSizeLists();
    return true;
  }

  void UpdateOp() override {
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                               "cudaStreamSynchronized failed");
    std::vector<TypeId> type_ids;
    std::vector<std::vector<size_t>> shapes;
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node_.lock());
    for (size_t i = 0; i < output_num; ++i) {
      std::vector<size_t> shape = common::AnfAlgo::GetOutputInferShape(kernel_node_.lock(), i);
      if (i == 0) {
        shape[0] = post_output_size_;
      }
      TypeId type_id = common::AnfAlgo::GetOutputInferDataType(kernel_node_.lock(), i);
      type_ids.emplace_back(type_id);
      shapes.emplace_back(shape);
    }
    common::AnfAlgo::SetOutputInferTypeAndShape(type_ids, shapes, kernel_node_.lock().get());
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    num_elements_ = 1;
    post_output_size_ = 0;
    is_null_input_ = false;
    stream_ptr_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(num_elements_ * sizeof(S));
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);
  }

 private:
  void *stream_ptr_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  int num_elements_;
  int post_output_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_UNIQUE_GPU_KERNEL_H_
