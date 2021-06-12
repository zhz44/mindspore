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

#include "ops/grad/hsigmoid_grad.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto grads_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("grads_shape", grads_shape, kEqual, "input_x_shape", input_x_shape, prim_name, TypeError);
  return std::make_shared<abstract::Shape>(grads_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 2, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  types.emplace("grads", input_args[0]->BuildType());
  types.emplace("input_x", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

AbstractBasePtr HSigmoidGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_EVAL_IMPL(HSigmoidGrad, prim::kPrimHSigmoidGrad, HSigmoidGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
