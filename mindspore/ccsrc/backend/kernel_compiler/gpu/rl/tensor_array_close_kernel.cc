/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/rl/tensor_array_close_kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/gpu/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::gpu::GPUTensorArray;
using mindspore::device::gpu::GPUTensorArrayPtr;
TensorArrayCloseKernel::TensorArrayCloseKernel() {}

const std::vector<size_t> &TensorArrayCloseKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &TensorArrayCloseKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &TensorArrayCloseKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool TensorArrayCloseKernel::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  InitSizeLists();
  return true;
}

void TensorArrayCloseKernel::InitSizeLists() {
  input_size_list_.push_back(sizeof(int64_t));
  output_size_list_.push_back(sizeof(int64_t));
}

bool TensorArrayCloseKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<AddressPtr> &, void *) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  GPUTensorArrayPtr tensors_ =
    std::dynamic_pointer_cast<GPUTensorArray>(TensorArrayMgr::GetInstance().GetTensorArray(handle_addr));
  MS_ERROR_IF_NULL(tensors_);
  // Free device mem
  tensors_->Free();
  // Erase tensorarray
  if (!TensorArrayMgr::GetInstance().EraseTensorArray(handle_addr)) {
    MS_LOG(EXCEPTION) << "Free tensorarray failed";
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
