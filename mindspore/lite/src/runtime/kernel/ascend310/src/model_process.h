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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_ACL_MODEL_PROCESS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_ACL_MODEL_PROCESS_H_

#include <vector>
#include <string>
#include <map>
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "include/api/types.h"
#include "include/errorcode.h"

using mindspore::lite::STATUS;

namespace mindspore {
namespace acl {
struct AclTensorInfo {
  void *cur_device_data;
  void *device_data;
  size_t buffer_size;
  aclDataType data_type;
  std::vector<int64_t> dims;
  std::string name;
};

class ModelProcess {
 public:
  ModelProcess()
      : model_id_(0xffffffff),
        is_run_on_device_(false),
        model_desc_(nullptr),
        inputs_(nullptr),
        outputs_(nullptr),
        input_infos_(),
        output_infos_() {}
  ~ModelProcess() {}

  STATUS UnLoad();
  STATUS PredictFromHost(const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs);
  STATUS PreInitModelResource();

  // override this method to avoid request/reply data copy
  void SetIsDevice(bool is_device) { is_run_on_device_ = is_device; }

  void set_model_id(uint32_t model_id) { model_id_ = model_id; }
  uint32_t model_id() const { return model_id_; }

 private:
  STATUS CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset);
  STATUS CheckAndInitInput(const std::vector<mindspore::MSTensor> &inputs);
  STATUS SortTensorInfoByName(const std::vector<mindspore::MSTensor> &tensor, std::vector<AclTensorInfo> *tensor_info);
  STATUS CheckTensorByTensorInfo(const std::vector<mindspore::MSTensor> &tensor,
                                 const std::vector<AclTensorInfo> &tensor_info, size_t dynamic_nums);
  STATUS GetOutputs(std::vector<mindspore::MSTensor> *outputs);
  STATUS ConstructTensor(std::vector<mindspore::MSTensor> *outputs);
  STATUS SetBatchSize(const std::vector<mindspore::MSTensor> &inputs);
  STATUS InitInputsBuffer();
  STATUS InitOutputsBuffer();
  STATUS ResetOutputSize();
  size_t GetDynamicDims(const std::vector<AclTensorInfo> &);
  STATUS ProcDynamicShape(const std::vector<mindspore::MSTensor> &inputs, size_t dynamic_nums);

  void DestroyInputsDataset();
  void DestroyInputsDataMem();
  void DestroyInputsBuffer();
  void DestroyOutputsBuffer();

  uint32_t model_id_;
  // if run one device(AICPU), there is no need to alloc device memory and copy inputs to(/outputs from) device
  bool is_run_on_device_;
  aclmdlDesc *model_desc_;
  aclmdlDataset *inputs_;
  aclmdlDataset *outputs_;
  std::vector<AclTensorInfo> input_infos_;
  std::vector<AclTensorInfo> output_infos_;
};
}  // namespace acl
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_ACL_MODEL_PROCESS_H_
