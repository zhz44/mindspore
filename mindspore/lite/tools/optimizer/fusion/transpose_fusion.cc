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

#include "tools/optimizer/fusion/transpose_fusion.h"
#include <unordered_map>
#include <memory>
#include <vector>
#include "tools/converter/quant_param_holder.h"
#include "mindspore/core/ops/transpose.h"
#include "tools/optimizer/common/format_utils.h"
#include "ops/fusion/scale_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
bool IsBNCNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimBatchNorm) ||
           CheckPrimitiveType(anf_node, prim::kPrimFusedBatchNorm);
  }
  return false;
}

VectorRef TransposeFusion::DefineBNPattern() const {
  auto is_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(transpose_param != nullptr, {});
  VectorRef transpose_conv_ref = VectorRef({is_transpose, is_conv, transpose_param});
  auto is_bn = std::make_shared<CondVar>(IsBNCNode);
  MS_CHECK_TRUE_RET(is_bn != nullptr, {});
  auto bn_mean_var = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bn_mean_var != nullptr, {});
  auto bn_variable_var = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bn_variable_var != nullptr, {});
  auto bn_other_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(bn_other_var != nullptr, {});
  VectorRef bn_ref = VectorRef({is_bn, transpose_conv_ref, bn_mean_var, bn_variable_var, bn_other_var});
  return bn_ref;
}

VectorRef TransposeFusion::DefineActivationscalePattern() const {
  auto is_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(transpose_param != nullptr, {});
  VectorRef transpose_conv_ref = VectorRef({is_transpose, is_conv, transpose_param});
  auto is_scale = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimScaleFusion>);
  MS_CHECK_TRUE_RET(is_scale != nullptr, {});
  auto scale_var_1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(scale_var_1 != nullptr, {});
  auto scale_var_2 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(scale_var_2 != nullptr, {});
  VectorRef sclae_ref = VectorRef({is_scale, transpose_conv_ref, scale_var_1, scale_var_2});
  return sclae_ref;
}

VectorRef TransposeFusion::DefineActivationPattern() const {
  auto is_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(transpose_param != nullptr, {});
  VectorRef transpose_conv_ref = VectorRef({is_transpose, is_conv, transpose_param});
  auto is_activation = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_activation != nullptr, {});
  VectorRef act_ref = VectorRef({is_activation, transpose_conv_ref});
  return act_ref;
}

VectorRef TransposeFusion::DefineBiasAddPattern() const {
  auto is_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(transpose_param != nullptr, {});
  VectorRef transpose_conv_ref = VectorRef({is_transpose, is_conv, transpose_param});
  auto is_bias_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBiasAdd>);
  MS_CHECK_TRUE_RET(is_bias_add != nullptr, {});
  auto bias_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bias_param != nullptr, {});
  VectorRef act_ref = VectorRef({is_bias_add, transpose_conv_ref, bias_param});
  return act_ref;
}

VectorRef TransposeFusion::DefineScalePattern() const {
  auto is_transpose = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto is_scale = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimScaleFusion>);
  MS_CHECK_TRUE_RET(is_scale != nullptr, {});
  auto is_weight_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_weight_param != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  VectorRef trans_trans_ref = VectorRef({is_scale, is_transpose, is_weight_param, is_seq_var});
  return trans_trans_ref;
}

VectorRef TransposeFusion::DefineTransTransPattern() const {
  auto is_transpose1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose1 != nullptr, {});
  auto is_transpose2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTranspose>);
  MS_CHECK_TRUE_RET(is_transpose2 != nullptr, {});
  auto transpose_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(transpose_param != nullptr, {});
  VectorRef trans_trans_ref = VectorRef({is_transpose2, is_transpose1, transpose_param});
  return trans_trans_ref;
}

std::unordered_map<std::string, VectorRef> TransposeFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns["BNPatternName"] = DefineBNPattern();
  patterns["ActivationPatternName"] = DefineActivationPattern();
  patterns["BiasAddPatternName"] = DefineBiasAddPattern();
  patterns["ScalePatternName"] = DefineActivationscalePattern();
  patterns["TransScalePatternName"] = DefineScalePattern();
  patterns["TransTransPatternName"] = DefineTransTransPattern();
  return patterns;
}

CNodePtr GenTransposeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const AnfNodePtr &perm,
                          const std::string &cnode_name) {
  MS_ASSERT(func_graph != nullptr && input_node != nullptr);
  auto trans_prim = std::make_shared<ops::Transpose>();
  MS_CHECK_TRUE_RET(trans_prim != nullptr, nullptr);
  auto cnode = func_graph->NewCNode(trans_prim, {input_node, perm});
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  cnode->set_fullname_with_scope(cnode_name);
  auto quant_params_holder = std::make_shared<lite::QuantParamHolder>(2, 1);
  MS_CHECK_TRUE_RET(quant_params_holder != nullptr, nullptr);
  auto trans_insert_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_ASSERT(trans_insert_prim != nullptr);
  trans_insert_prim->AddAttr("quant_params", quant_params_holder);
  return cnode;
}

AnfNodePtr TransposeFusion::TransTransFusion(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  auto trans_cnode_2 = node->cast<CNodePtr>();
  if (IsMarkedTrainOp(trans_cnode_2)) {
    return nullptr;
  }
  MS_CHECK_TRUE_RET(trans_cnode_2 != nullptr, nullptr);
  if (!CheckPrimitiveType(trans_cnode_2, prim::kPrimTranspose) ||
      !CheckPrimitiveType(trans_cnode_2->input(1), prim::kPrimTranspose)) {
    return nullptr;
  }
  std::vector<int> post_perm;
  if (GetTransposePerm(trans_cnode_2, &post_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "get transpose perm failed.";
    return nullptr;
  }
  std::vector<int> pre_perm;
  auto pre_node = trans_cnode_2->input(1);
  auto pre_cnode = pre_node->cast<CNodePtr>();
  if (pre_cnode == nullptr) {
    return nullptr;
  }
  if (IsMarkedTrainOp(pre_cnode)) {
    return nullptr;
  }
  if (GetTransposePerm(pre_cnode, &pre_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "get transpose perm failed.";
    return nullptr;
  }
  if ((pre_perm == kNH2NC && post_perm == kNC2NH) || (pre_perm == kNC2NH && post_perm == kNH2NC)) {
    return pre_cnode->input(1);
  }
  if (pre_perm.size() == post_perm.size()) {
    std::vector<int> perm;
    for (auto idx : post_perm) {
      MS_CHECK_TRUE_RET(idx >= 0 && static_cast<size_t>(idx) < pre_perm.size(), nullptr);
      perm.push_back(pre_perm[idx]);
    }
    std::vector<int> ori_perm = [&pre_perm]() {
      std::vector<int> value;
      for (int i = 0; i < static_cast<int>(pre_perm.size()); i++) value.push_back(i);
      return value;
    }();
    if (perm == ori_perm) {
      return pre_cnode->input(1);
    }
    auto name = pre_cnode->fullname_with_scope();
    auto new_transpose = GenTransposeNode(func_graph, pre_cnode->input(1), perm, name);
    auto manager = func_graph->manager();
    MS_ASSERT(manager != nullptr);
    manager->Replace(trans_cnode_2, new_transpose);
    return new_transpose;
  }
  return nullptr;
}

int TransposeFusion::AdjustAxisOfScale(const mindspore::AnfNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  auto scale_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(scale_cnode != nullptr, lite::RET_ERROR);
  if (IsMarkedTrainOp(scale_cnode)) {
    return lite::RET_ERROR;
  }
  auto transpose_node = scale_cnode->input(1);
  auto transpose_cnode = transpose_node->cast<CNodePtr>();
  if (transpose_cnode == nullptr) {
    return lite::RET_ERROR;
  }
  if (IsMarkedTrainOp(transpose_cnode)) {
    return lite::RET_ERROR;
  }

  std::vector<int> perm;
  if (GetTransposePerm(transpose_cnode, &perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "get tanspose perm failed.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(!perm.empty(), lite::RET_ERROR);
  auto scale_prim = GetValueNode<std::shared_ptr<ops::ScaleFusion>>(scale_cnode->input(0));
  MS_CHECK_TRUE_RET(scale_prim != nullptr, lite::RET_ERROR);
  MS_CHECK_TRUE_RET(scale_prim->GetAttr(ops::kAxis) != nullptr, lite::RET_ERROR);
  int axis = scale_prim->get_axis() < 0 ? scale_prim->get_axis() + perm.size() : scale_prim->get_axis();
  auto weight_param = scale_cnode->input(2);
  MS_CHECK_TRUE_RET(weight_param != nullptr, lite::RET_ERROR);
  std::vector<int64_t> weight_shape;
  if (FetchShapeFromAbstract(weight_param->abstract(), &weight_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape from abstract failed.";
    return lite::RET_ERROR;
  }
  if (weight_shape.size() != 1) {
    MS_LOG(ERROR) << "Dot not support weight size larger than 1.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(axis >= 0 && static_cast<size_t>(axis) < perm.size(), lite::RET_ERROR);
  scale_prim->set_axis(static_cast<int64_t>(perm.at(axis)));
  if (perm == kNC2NH) {
    scale_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NCHW));
  } else if (perm == kNH2NC) {
    scale_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
  }
  return lite::RET_OK;
}

AnfNodePtr TransposeFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                    const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (pattern_name == "TransTransPatternName") {
    return TransTransFusion(func_graph, node);
  } else if (pattern_name == "TransScalePatternName" && AdjustAxisOfScale(node) != lite::RET_OK) {
    return nullptr;
  }

  if (node->cast<CNodePtr>() == nullptr) {
    return nullptr;
  }
  auto any_cnode = node->cast<CNodePtr>();
  if (IsMarkedTrainOp(any_cnode)) {
    return nullptr;
  }
  const auto transpose_node = any_cnode->input(1);
  if (transpose_node == nullptr || transpose_node->cast<CNodePtr>() == nullptr) {
    return nullptr;
  }
  const CNodePtr &transpose_cnode = transpose_node->cast<CNodePtr>();
  if (IsMarkedTrainOp(transpose_cnode)) {
    return nullptr;
  }
  auto perm_node = transpose_cnode->input(kInputIndexTwo);
  auto trans_post_node = GenTransposeNode(func_graph, any_cnode, perm_node, any_cnode->fullname_with_scope() + "_post");
  MS_CHECK_TRUE_RET(trans_post_node != nullptr, nullptr);
  if (any_cnode->abstract() != nullptr) {
    trans_post_node->set_abstract(any_cnode->abstract()->Clone());
  }
  if (transpose_cnode->input(1)->abstract() != nullptr) {
    any_cnode->set_abstract(transpose_cnode->input(1)->abstract()->Clone());
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(any_cnode, 1, transpose_cnode->input(1));
  return trans_post_node;
}
}  // namespace mindspore::opt
