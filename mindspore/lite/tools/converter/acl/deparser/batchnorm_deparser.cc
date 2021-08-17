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

#include "tools/converter/acl/deparser/batchnorm_deparser.h"
#include <memory>
#include "tools/converter/acl/deparser/primitive_deparser_register.h"
#include "tools/converter/acl/deparser/tbe_op_def.h"
#include "include/registry/parser_context.h"

namespace mindspore {
namespace lite {
STATUS BatchNormDeparser::Deparser(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  auto attr_val = src_prim->GetAttr(ops::kFmkType);
  int fmk_type = attr_val != nullptr ? GetValue<int>(attr_val) : converter::kFmkTypeTf;
  if (fmk_type == converter::kFmkTypeCaffe) {
    auto dst_prim = std::make_shared<acl::BNInference>();
    if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
      MS_LOG(ERROR) << "BatchNorm deparser failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_DEPARSER(kNameBatchNorm, BatchNormDeparser)
}  // namespace lite
}  // namespace mindspore
