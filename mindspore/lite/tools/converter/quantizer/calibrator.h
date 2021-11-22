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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CALIBRATOR_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CALIBRATOR_H
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/diverg_info.h"

namespace mindspore::lite::quant {
class Calibrator {
 public:
  explicit Calibrator(size_t bit_num, int quant_max, int quant_min)
      : bit_num_(bit_num), quant_max_(quant_max), quant_min_(quant_min) {}

  ~Calibrator() = default;

  int GenerateInputData(const std::string &input_name, size_t image_index, mindspore::tensor::MSTensor *tensor) const;

  size_t GetBatchNum() const { return data_pre_process_param_.calibrate_size; }

  uint32_t GetThreadNum() const { return thread_; }

  bool GetBiasCorrection() const { return full_quant_param_.bias_correction; }

  size_t GetInputNum() const { return data_pre_process_param_.calibrate_path_vector.size(); }

  int AddQuantizedOp(const CNodePtr &cnode);

  int RecordMaxMinValue(const std::vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info);

  int UpdateDivergInterval(std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *diverg_info);

  int UpdateDataFrequency(const std::vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info);

  int ComputeThreshold();

  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *GetInputDivergInfo();

  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *GetOutputDivergInfo();

  FullQuantParam full_quant_param_;

  preprocess::DataPreProcessParam data_pre_process_param_;

  int thread_ = 4;

 private:
  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> inputs_diverg_info_;

  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> outputs_diverg_info_;

  size_t bit_num_;
  int quant_max_;
  int quant_min_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER__CALIBRATOR_H