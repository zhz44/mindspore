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

#include "backend/optimizer/ascend/enhancer/insert_depend_for_all_gather.h"
#include <memory>
#include <map>
#include "utils/utils.h"
#include "backend/optimizer/ascend/ascend_helper.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
bool InsertDependForAllGather::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::map<int64_t, AnfNodePtr> all_gather_node;
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfAlgo::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::GetCNodeName(cnode) == kAllGatherOpName && AnfAlgo::HasNodeAttr(kAttrFusion, cnode) &&
        AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion) > 0) {
      all_gather_node[AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion)] = node;
    }
  }
  auto iter = all_gather_node.begin();
  for (int64_t i = 0; i < SizeToInt(all_gather_node.size()) - 1; ++i) {
    auto current_node = iter->second;
    auto next_node = (++iter)->second;
    auto next_cnode = next_node->cast<CNodePtr>();
    std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                      AnfAlgo::GetInputNode(next_cnode, 0), current_node};
    auto new_input = graph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(new_input);
    new_input->set_abstract(AnfAlgo::GetInputNode(next_cnode, 0)->abstract());
    AnfAlgo::SetNodeInput(next_cnode, new_input, 0);
    changed = true;
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
