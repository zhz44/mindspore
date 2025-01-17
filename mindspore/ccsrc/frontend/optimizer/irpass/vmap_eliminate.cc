/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/vmap_eliminate.h"
#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "pipeline/pynative/pynative_execute.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
// White list of primitives consistent before and after transformation.
mindspore::HashSet<std::string> throughtout_op{prim::kPrimMakeTuple->name(), prim::kPrimMakeList->name(),
                                               prim::kPrimDepend->name(), prim::kPrimReturn->name(),
                                               prim::kPrimUpdateState->name()};

AnfNodePtr BindInAxis(const CNodePtr &vmap_app, const pipeline::ResourceBasePtr &resource, const ValuePtr &in_axes) {
  FuncGraphPtr vmap_fg = vmap_app->func_graph();
  FuncGraphPtr bind_axes = nullptr;
  py::function fn;
  constexpr char get_fn[] = "bind_in_axes";
  constexpr char module[] = "mindspore.ops._vmap";
  fn = python_adapter::GetPyFn(module, get_fn);
  bind_axes = parse::ParsePythonCode(fn);
  MS_EXCEPTION_IF_NULL(bind_axes);
  pipeline::ResourceBasePtr res = (resource != nullptr) ? resource : std::make_shared<pipeline::Resource>();
  (void)parse::ResolveFuncGraph(bind_axes, res);

  std::vector<AnfNodePtr> inputs;
  auto make_tuple = NewValueNode(prim::kPrimMakeTuple);
  inputs.push_back(make_tuple);
  int vmap_app_inputs_size = SizeToInt(vmap_app->size());
  int monad_index = -1;
  for (int i = 1; i < vmap_app_inputs_size; ++i) {
    AnfNodePtr para_of_ori_graph = vmap_app->input(i);
    // Drop the side effect tag parameters, because it has no mapping axis.
    if (HasAbstractMonad(para_of_ori_graph)) {
      monad_index = i;
      break;
    }
    inputs.push_back(para_of_ori_graph);
  }
  auto inputs_node = vmap_fg->NewCNode(inputs);
  auto bind_axes_node = vmap_fg->NewCNode({NewValueNode(bind_axes), inputs_node, NewValueNode(in_axes)});

  std::vector<AnfNodePtr> bind_axes_vec;
  bind_axes_vec.push_back(vmap_app->input(0));

  int para_size = (monad_index != -1) ? monad_index - 1 : vmap_app_inputs_size - 1;

  for (int64_t i = 0; i < para_size; ++i) {
    std::vector<AnfNodePtr> each_arg;
    auto get_item = NewValueNode(prim::kPrimTupleGetItem);
    each_arg.push_back(get_item);
    each_arg.push_back(bind_axes_node);
    each_arg.push_back(NewValueNode(i));
    auto each_arg_node = vmap_fg->NewCNode(each_arg);
    bind_axes_vec.push_back(each_arg_node);
  }
  if (monad_index != -1) {
    for (int i = monad_index; i < vmap_app_inputs_size; ++i) {
      bind_axes_vec.push_back(vmap_app->input(i));
    }
  }

  auto bind_in_axes_node = vmap_fg->NewCNode(bind_axes_vec);
  return bind_in_axes_node;
}

int GetAxisSizeByAbs(const AbstractBasePtr &abs, const ValuePtr &in_axes) {
  MS_EXCEPTION_IF_NULL(abs);
  MS_EXCEPTION_IF_NULL(in_axes);
  int axis_size = -1;
  auto abs_sequence = dyn_cast<abstract::AbstractSequence>(abs);
  if (abs_sequence != nullptr) {
    AbstractBasePtrList abs_list = abs_sequence->elements();
    auto in_axes_seq = dyn_cast<ValueSequeue>(in_axes);
    int index = 0;
    for (auto sub_abs : abs_list) {
      ValuePtr sub_in_axes = in_axes;
      if (in_axes->isa<ValueSequeue>()) {
        sub_in_axes = (*in_axes_seq)[index];
        index++;
      }
      axis_size = GetAxisSizeByAbs(sub_abs, sub_in_axes);
      if (axis_size != -1) {
        return axis_size;
      }
    }
  }

  auto in_axes_int = dyn_cast<Int64Imm>(in_axes);
  if (in_axes_int != nullptr) {
    int axis = in_axes_int->value();
    ShapeVector orig_shape = dyn_cast<abstract::Shape>(abs->BuildShape())->shape();
    int shape_len = SizeToInt(orig_shape.size());
    if (axis < -shape_len || axis >= shape_len) {
      MS_LOG(EXCEPTION) << "ValueError: axis " << axis << " is out of bounds for array of dimension [" << -shape_len
                        << "," << shape_len << ").";
    }
    axis = axis < 0 ? shape_len + axis : axis;
    axis_size = orig_shape[axis];
    return axis_size;
  }
  return axis_size;
}

int GetAxisSize(const ValuePtr &in_axes, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  // `axis_size` is unique within the scope of vmap, so we just need to get one of them.
  int axis_size = -1;
  auto in_axes_seq = dyn_cast<ValueSequeue>(in_axes);
  size_t parameters_size = cnode->size() - 1;
  for (size_t i = 0; i < parameters_size; ++i) {
    ValuePtr sub_in_axes = (in_axes->isa<ValueSequeue>()) ? (*in_axes_seq)[i] : in_axes;
    auto sub_abs = cnode->input(i + 1)->abstract();
    axis_size = GetAxisSizeByAbs(sub_abs, sub_in_axes);
    if (axis_size != -1) {
      return axis_size;
    }
  }
  return axis_size;
}

AnfNodePtr MatchOutAxis(const AnfNodePtr &expanded_vmap_node, int parameters_size, int axis_size,
                        const pipeline::ResourceBasePtr &resource, const ValuePtr &out_axes) {
  FuncGraphPtr vmap_post_fg = std::make_shared<FuncGraph>();

  FuncGraphPtr match_out_axis = nullptr;
  py::function fn;
  constexpr char get_fn[] = "match_out_axis";
  constexpr char module[] = "mindspore.ops._vmap";
  fn = python_adapter::GetPyFn(module, get_fn);
  match_out_axis = parse::ParsePythonCode(fn);
  MS_EXCEPTION_IF_NULL(match_out_axis);
  pipeline::ResourceBasePtr res = (resource != nullptr) ? resource : std::make_shared<pipeline::Resource>();
  (void)parse::ResolveFuncGraph(match_out_axis, res);

  std::vector<AnfNodePtr> exec_node;
  exec_node.push_back(expanded_vmap_node);
  for (int i = 0; i < parameters_size; ++i) {
    exec_node.push_back(vmap_post_fg->add_parameter());
  }
  auto vmap_outputs = vmap_post_fg->NewCNode(exec_node);
  auto match_out_axis_app = vmap_post_fg->NewCNode({NewValueNode(match_out_axis), vmap_outputs, NewValueNode(out_axes),
                                                    NewValueNode(static_cast<int64_t>(axis_size))});
  vmap_post_fg->set_output(match_out_axis_app);

  return NewValueNode(vmap_post_fg);
}

FuncGraphPtr GetVmapRule(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resource, int axis_size) {
  // Set a child scope named "vmap_'PrimitiveName'" for the vmap rule function,
  // and add "VmapRule" to the front.
  constexpr char vmap_rule_scope[] = "VmapRule/";
  constexpr char vmap_op_child_scope_prefix[] = "/vmap_";
  MS_EXCEPTION_IF_NULL(prim);
  auto scope = std::make_shared<Scope>(vmap_rule_scope + ScopeManager::GetInstance().GetCurrentScope()->name() +
                                       vmap_op_child_scope_prefix + prim->name());
  ScopeGuard scope_guard(scope);

  // Firstly we parse the python VmapRules function registered for specific primitive. If failed, get
  // the vmap general rule.
  FuncGraphPtr vmap_rule_fg = nullptr;
  py::function vmap_rule_fn;
  bool is_side_effect = false;
  if (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM)) {
    is_side_effect = true;
  } else if (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_IO) && prim->name() != prim::kPrimPrint->name()) {
    MS_LOG(EXCEPTION) << prim->name() << " is a GRAPH_FLAG_SIDE_EFFECT_IO prim, vmap dont support currently.";
  }
  if (prim->is_base()) {
    vmap_rule_fn = GetVmapRuleFunction(prim->name(), axis_size);
  } else {
    vmap_rule_fn = prim->cast<PrimitivePyPtr>()->GetVmapRuleFunction(is_side_effect, axis_size);
    if (py::isinstance<py::none>(vmap_rule_fn)) {
      vmap_rule_fn = GetVmapRuleFunction(prim->name(), axis_size);
    }
  }
  if (!vmap_rule_fn || py::isinstance<py::none>(vmap_rule_fn)) {
    MS_LOG(DEBUG) << "Fail to find vmap rule function for " << prim->name() << ", try to get the general vmap rule.";
    vmap_rule_fn = GetVmapGeneralRuleFunction(prim->name(), is_side_effect, axis_size);
  }
  if (!vmap_rule_fn || py::isinstance<py::none>(vmap_rule_fn)) {
    MS_LOG(EXCEPTION) << "Fail to find vmap rule function for " << prim->name() << ".";
  }
  vmap_rule_fg = parse::ParsePythonCode(vmap_rule_fn);
  if (vmap_rule_fg == nullptr) {
    MS_LOG(EXCEPTION) << "Fail to parse vmap rule function for " << prim->name() << ".";
  }
  auto vmap_rule_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_PROPAGATE);
  if (vmap_rule_flag) {
    vmap_rule_fg->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  }
  pipeline::ResourceBasePtr res = (resource != nullptr) ? resource : std::make_shared<pipeline::Resource>();
  (void)parse::ResolveFuncGraph(vmap_rule_fg, res);

  return vmap_rule_fg;
}

AnfNodePtr ExpandVmapPrimitive(const AnfNodePtr &vnode, const pipeline::ResourceBasePtr &resource, int axis_size) {
  MS_EXCEPTION_IF_NULL(vnode);
  if (!IsValueNode<Primitive>(vnode)) {
    MS_LOG(EXCEPTION) << "Primitive node is not valid.";
  }
  auto prim = GetValueNode<PrimitivePtr>(vnode);
  MS_LOG(DEBUG) << "Overloading Primitive node " << vnode->DebugString() << ".";
  if (throughtout_op.count(prim->name())) {
    return vnode;
  } else {
    FuncGraphPtr prim_vmap_rule = GetVmapRule(prim, resource, axis_size);
    if (prim_vmap_rule == nullptr) {
      MS_LOG(EXCEPTION) << "Primitive " << prim->name() << " transform to VmapRule failed. NodeInfo: "
                        << trace::GetDebugInfo(prim_vmap_rule->debug_info()) << ".";
    }
    return NewValueNode(prim_vmap_rule);
  }
  return nullptr;
}

void BindNoneAxis(const AnfNodePtr &node, const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng) {
  MS_EXCEPTION_IF_NULL(node);
  const auto node_user_map = mng->node_users();
  auto user = node_user_map.find(node);
  if (user != node_user_map.end() && !user->second.empty()) {
    auto make_tuple = NewValueNode(prim::kPrimMakeTuple);
    auto replace_node = func_graph->NewCNode({make_tuple, node, NewValueNode(kNone)});
    for (auto pair : user->second) {
      if (pair.first->func_graph() == func_graph) {
        auto user_node = pair.first->cast<CNodePtr>();
        mng->SetEdge(user_node, pair.second, replace_node);
      }
    }
  }
}

void ExpandVmapValueNode(const FuncGraphPtr &vmap_fg, const pipeline::ResourceBasePtr &resource,
                         mindspore::HashSet<AnfNodePtr> *visited_node, int axis_size) {
  // Map ValueNode.
  auto manager = resource->manager();
  auto value_nodes = vmap_fg->value_nodes();
  for (const auto &value_pair : value_nodes) {
    auto node = value_pair.first;
    // ValueNode may have been transformed when other graphs are expanded.
    if (visited_node->count(node)) {
      MS_LOG(DEBUG) << node->DebugString() << " has been transformed.";
      continue;
    }
    if (IsValueNode<FuncGraph>(node)) {
      MS_LOG(DEBUG) << "Map FuncGraph node " << node->DebugString() << ".";
      visited_node->insert(node);
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(node);
      auto transformed_fg = ExpandVmapFunctor(sub_func_graph, resource, visited_node, axis_size);
      auto replace_node = NewValueNode(transformed_fg);
      visited_node->insert(replace_node);
      manager->Replace(node, replace_node);
    } else if (IsValueNode<Primitive>(node)) {
      auto replace_node = ExpandVmapPrimitive(node, resource, axis_size);
      MS_EXCEPTION_IF_NULL(replace_node);
      visited_node->insert(replace_node);
      manager->Replace(node, replace_node);
    } else if (IsValueNode<Scalar>(node) || IsValueNode<tensor::Tensor>(node) || IsValueNode<None>(node) ||
               IsValueNode<ValueTuple>(node) || IsValueNode<Type>(node) || IsValueNode<StringImm>(node)) {
      auto value_node_ptr = node->cast<ValueNodePtr>();
      ValuePtr node_value = value_node_ptr->value();
      std::vector<ValuePtr> elements;
      elements.push_back(node_value);
      elements.push_back(kNone);
      auto replace_value = std::make_shared<ValueTuple>(elements);
      auto replace_node = NewValueNode(replace_value);
      visited_node->insert(replace_node);
      manager->Replace(node, replace_node);
    } else if (IsValueNode<Monad>(node)) {
      continue;
    } else {
      MS_LOG(EXCEPTION) << "vmap do not support transform " << node->DebugString() << " right now.";
    }
  }
}

void ExpandVmapFreeVariable(const FuncGraphPtr &vmap_fg, const FuncGraphManagerPtr &manager,
                            const mindspore::HashSet<AnfNodePtr> &visited_node) {
  // Map free variable.
  auto free_variables_nodes = vmap_fg->free_variables_nodes();
  for (auto &node : free_variables_nodes) {
    if (visited_node.count(node) || node->isa<CNode>()) {
      MS_LOG(DEBUG) << node->DebugString() << " has been transformed.";
    } else if (node->isa<Parameter>() || IsValueNode<Scalar>(node) || IsValueNode<tensor::Tensor>(node) ||
               IsValueNode<None>(node) || IsValueNode<ValueTuple>(node) || IsValueNode<Type>(node)) {
      BindNoneAxis(node, vmap_fg, manager);
    } else {
      MS_LOG(EXCEPTION) << "vmap do not support transform " << node->DebugString() << " right now.";
    }
  }
}

FuncGraphPtr ExpandVmapFunctor(const FuncGraphPtr &vmap_fg, const pipeline::ResourceBasePtr &resource,
                               mindspore::HashSet<AnfNodePtr> *visited_node, int axis_size) {
  MS_EXCEPTION_IF_NULL(vmap_fg);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(vmap_fg);

  // The parameters of the current graph will be transformed in the upper graph, and recorded in
  // `visited_node` to avoid being repeatedly transformed refer as a free variable in other graph.
  auto parameter_nodes = vmap_fg->parameters();
  for (auto &node : parameter_nodes) {
    MS_LOG(DEBUG) << "parameter_nodes" << node->DebugString() << ".";
    visited_node->insert(node);
  }

  ExpandVmapValueNode(vmap_fg, resource, visited_node, axis_size);
  ExpandVmapFreeVariable(vmap_fg, manager, *visited_node);

  return vmap_fg;
}

// Entry to perform Vmap transformation.
AnfNodePtr ExpandVmap(const ValueNodePtr &vnode, const pipeline::ResourceBasePtr &resource, int axis_size) {
  MS_EXCEPTION_IF_NULL(vnode);
  if (IsValueNode<FuncGraph>(vnode)) {
    ScopeGuard scope_guard(vnode->scope());
    auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_LOG(DEBUG) << "Funcgraph: " << func_graph->ToString() << " will perform the Vmap transformation.";

    // Record transformed FuncGraphs and other nodes to avoid repeatedly expanding and transforming.
    // Whose lifecycle is limited to the current extension.
    mindspore::HashSet<AnfNodePtr> visited_node;
    auto tf_fg = ExpandVmapFunctor(func_graph, resource, &visited_node, axis_size);
    visited_node.clear();

    return NewValueNode(tf_fg);
  }
  MS_LOG(EXCEPTION) << "Currently, the first argument in F.vmap only supports Cell, Python defined "
                       "function or @ms_function decorated function.";
  return nullptr;
}
}  // namespace internal

bool ExpandVmapPrim::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  // Expand vmap nodes that don't have embed j or vmap nodes.
  bool change = false;
  auto manager = optimizer->manager();
  for (auto &vmap_node : prim_nodes_) {
    auto VmapPrim = GetValueNode<PrimitivePtr>(vmap_node->input(0));
    MS_EXCEPTION_IF_NULL(VmapPrim);
    ValuePtr in_axes = VmapPrim->GetAttr("in_axes");
    MS_EXCEPTION_IF_NULL(in_axes);
    ValuePtr out_axes = VmapPrim->GetAttr("out_axes");
    MS_EXCEPTION_IF_NULL(out_axes);

    auto vmap_fn_node = vmap_node->input(1);
    auto vmap_fg = GetValueNode<FuncGraphPtr>(vmap_fn_node);
    auto &fn_users = manager->node_users()[vmap_fn_node];
    size_t fn_users_size = fn_users.size();

    auto users = manager->node_users()[vmap_node];
    if (users.size() < 1) {
      MS_LOG(EXCEPTION) << "vmap_node could used by at least one CNode, but got users.size() = " << users.size() << ".";
    }
    size_t user_nb = 0;
    size_t user_size = users.size();
    for (auto &user : users) {
      user_nb++;

      // When `vmap_node` has more than one user or `fn` has more than one user, the original function graph
      // cannot be modified directly.
      if ((user_size > 1 && user_nb != user_size) || fn_users_size > 1) {
        MS_LOG(DEBUG) << "Funcgraph: " << vmap_fg->ToString() << " is also used outside the scope of vmap.";
        auto vmap_fg_copy = BasicClone(vmap_fg, true);
        auto manager_ptr = optimizer->resource()->manager();
        manager_ptr->AddFuncGraph(vmap_fg_copy);
        vmap_fn_node = NewValueNode(vmap_fg_copy);
      } else {
        vmap_fn_node = NewValueNode(vmap_fg);
      }

      // get axis size
      auto vmap_app = user.first->cast<CNodePtr>();
      int user_index = user.second;
      int parameters_size = SizeToInt(vmap_app->size() - 1);
      int axis_size = internal::GetAxisSize(in_axes, vmap_app);
      if (axis_size == -1) {
        MS_LOG(EXCEPTION) << "Failed to get 'axis_size' within the scope of vmap.";
      }
      MS_LOG(DEBUG) << "The axis size corresponding to the current level vmap scope is " << axis_size << ".";

      // Step1: Bind the inputs with the corresponding in_axes.
      auto bind_axes_node = internal::BindInAxis(vmap_app, optimizer->resource(), in_axes);
      MS_EXCEPTION_IF_NULL(bind_axes_node);
      manager->Replace(vmap_app, bind_axes_node);

      // Step2: Bind the variables with the corresponding axis, and overload the original
      // operation with the VmapRule operation meanwhile transfer the axis information.
      auto expanded_vmap = internal::ExpandVmap(vmap_fn_node->cast<ValueNodePtr>(), optimizer->resource(), axis_size);
      MS_EXCEPTION_IF_NULL(expanded_vmap);

      // Step3: Convert the outputs according to the out_axes to the specified physical perspective.
      auto match_out_axis =
        internal::MatchOutAxis(expanded_vmap, parameters_size, axis_size, optimizer->resource(), out_axes);
      MS_EXCEPTION_IF_NULL(match_out_axis);
      manager->SetEdge(bind_axes_node, user_index, match_out_axis);
    }
    change = true;
  }
  return change;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
