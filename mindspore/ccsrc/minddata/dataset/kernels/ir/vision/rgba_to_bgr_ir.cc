/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/rgba_to_bgr_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/rgba_to_bgr_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RgbaToBgrOperation.
RgbaToBgrOperation::RgbaToBgrOperation() {}

RgbaToBgrOperation::~RgbaToBgrOperation() = default;

std::string RgbaToBgrOperation::Name() const { return kRgbaToBgrOperation; }

Status RgbaToBgrOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbaToBgrOperation::Build() {
  std::shared_ptr<RgbaToBgrOp> tensor_op = std::make_shared<RgbaToBgrOp>();
  return tensor_op;
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
